import os
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from torch.distributed.checkpoint import save as dist_cp_save
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2DecoderLayer,
    T5Gemma2EncoderLayer,
)
from config import LANG_PROMPTS
from utils import load_model_from_checkpoint, initialize_optimizer, initialize_scheduler, save_model




def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def orpo_preprocess_function(examples):
    prompts = [
        LANG_PROMPTS['ru'] + tox
        for tox in examples["toxic_text"]
    ]

    chosen = [
        text + tokenizer.eos_token
        for text in examples["chosen"]
    ]

    rejected = [
        text + tokenizer.eos_token
        for text in examples["rejected"]
    ]

    inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )

    chosen_tokens = tokenizer(
        chosen,
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

    rejected_tokens = tokenizer(
        rejected,
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

    chosen_labels = chosen_tokens["input_ids"]
    rejected_labels = rejected_tokens["input_ids"]

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
        "detox_comment": prompts, 
        "neutral_comment": chosen
    }



def orpo_collate_fn(features):
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    chosen_labels = [f["chosen_labels"] for f in features]
    rejected_labels = [f["rejected_labels"] for f in features]

    batch_inputs = tokenizer.pad(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        padding=True,
        return_tensors="pt",
    )

    chosen_batch = tokenizer.pad(
        {
            "input_ids": chosen_labels,
        },
        padding=True,
        return_tensors="pt",
    )

    rejected_batch = tokenizer.pad(
        {
            "input_ids": rejected_labels,
        },
        padding=True,
        return_tensors="pt",
    )

    chosen_labels = chosen_batch["input_ids"]
    rejected_labels = rejected_batch["input_ids"]

    chosen_labels[
        chosen_labels == tokenizer.pad_token_id
    ] = -100

    rejected_labels[
        rejected_labels == tokenizer.pad_token_id
    ] = -100

    chosen_len = (chosen_labels != -100).sum(dim=1)
    rejected_len = (rejected_labels != -100).sum(dim=1)

    return {
        "input_ids": batch_inputs["input_ids"],
        "attention_mask": batch_inputs["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
        "chosen_len": chosen_len,
        "rejected_len": rejected_len,
        "detox_comment": [
            f["detox_comment"]
            for f in features
        ],
        "neutral_comment": [
            f["neutral_comment"]
            for f in features
        ],
    }


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    x = x.float().clamp(max=-1e-6)       
    use_expm1 = x > -0.6931471805599453
    safe = torch.full_like(x, -1.0)
    x_a = torch.where(use_expm1, x, safe) 
    x_b = torch.where(use_expm1, safe, x) 
    
    return torch.where(
        use_expm1,
        torch.log(-torch.expm1(x_a)),
        torch.log1p(-torch.exp(x_b)),
    )


def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_n_tokens: torch.Tensor,
    rejected_n_tokens: torch.Tensor,
    orpo_lambda: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    chosen_mean = policy_chosen_logps / chosen_n_tokens.clamp(min=1)
    rejected_mean = policy_rejected_logps / rejected_n_tokens.clamp(min=1)

    log_odds = (chosen_mean - _log1mexp(chosen_mean)) - (rejected_mean - _log1mexp(rejected_mean))

    or_loss = -F.logsigmoid(log_odds).mean()
    nll = -chosen_mean.mean()
    loss = nll + orpo_lambda * or_loss
    
    return loss, chosen_mean.detach(), rejected_mean.detach()


def sequence_logprobs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)

    mask = labels != -100

    safe_labels = labels.masked_fill(~mask, 0)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * mask

    sequence_log_probs = token_log_probs.sum(dim=-1)
    n_tokens = mask.sum(dim=-1)

    return sequence_log_probs, n_tokens


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_steps = 0

    log_loss = 0.0
    log_steps = 0

    rank = dist.get_rank()

    for step, batch in enumerate(
        tqdm(
            train_loader,
            desc=f"Training rank={rank}",
            total=len(train_loader),
        )
    ):
        optimizer.zero_grad()

        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        chosen_decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
            batch["chosen_labels"]
        )
        rejected_decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
            batch["rejected_labels"]
        )
        
        output_chosen = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=chosen_decoder_input_ids,
        )

        output_rejected = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=rejected_decoder_input_ids,
        )

        logits_chosen = output_chosen["logits"]
        logits_rejected = output_rejected["logits"]

        chosen_logps, chosen_len = sequence_logprobs(
            logits_chosen,
            batch["chosen_labels"],
        )

        rejected_logps, rejected_len = sequence_logprobs(
            logits_rejected,
            batch["rejected_labels"],
        )

        loss, _, _ = orpo_loss(
            chosen_logps,
            rejected_logps,
            chosen_len,
            rejected_len,
            orpo_lambda=0.25
        )

        loss.backward()

        grad_norm = model.clip_grad_norm_(1.0)
        
        if rank == 0 and step >= 20:
            print(
                f"step={step + 1}, "
                f"loss={loss.item():.6f}, "
                f"grad_norm={grad_norm}"
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        global_loss = loss.detach().clone()
        dist.all_reduce(
            global_loss,
            op=dist.ReduceOp.AVG,
        )

        global_loss_value = global_loss.item()

        total_loss += global_loss_value
        total_steps += 1
        
        log_loss += global_loss_value
        log_steps += 1

        if rank == 0 and log_steps == 20:
            avg_loss = log_loss / log_steps

            print(
                f"Epoch {epoch}, "
                f"step {step + 1}/{len(train_loader)}, "
                f"loss={avg_loss:.4f}"
            )

            log_loss = 0.0
            log_steps = 0

    if rank == 0 and log_steps > 0:
        avg_loss = log_loss / log_steps

        print(
            f"Epoch {epoch}, "
            f"step {len(train_loader)}/{len(train_loader)}, "
            f"loss={avg_loss:.4f}"
        )

    epoch_loss = total_loss / max(total_steps, 1)
    return epoch_loss


def eval_epoch(
    model,
    test_loader,
    device,
):
    model.eval()
    rank = dist.get_rank()

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(
            test_loader,
            desc=f"Evaluating loss rank={rank}",
            total=len(test_loader),
        ):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            chosen_decoder_input_ids = (
                model.prepare_decoder_input_ids_from_labels(
                    batch["chosen_labels"]
                )
            )
            rejected_decoder_input_ids = (
                model.prepare_decoder_input_ids_from_labels(
                    batch["rejected_labels"]
                )
            )

            output_chosen = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=chosen_decoder_input_ids,
            )
            output_rejected = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=rejected_decoder_input_ids,
            )
            
            chosen_logps, chosen_len = sequence_logprobs(
                output_chosen["logits"],
                batch["chosen_labels"],
            )
            rejected_logps, rejected_len = sequence_logprobs(
                output_rejected["logits"],
                batch["rejected_labels"],
            )

            loss, _, _ = orpo_loss(
                chosen_logps,
                rejected_logps,
                chosen_len,
                rejected_len,
                orpo_lambda=0.25
            )
            total_loss += loss.item()
            total_steps += 1

    local_loss = total_loss / max(total_steps, 1)
    global_loss = torch.tensor(local_loss, dtype=torch.float32, device=device)
    
    dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
    return global_loss.item()



def save_model(
    checkpoint_dir,
    model,
    tokenizer,
    best,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = ("model_best" if best else "model_latest")
    checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint_name,
    )

    dist.barrier()

    save_policy = ShardedStateDictConfig(offload_to_cpu=False)
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        save_policy,
    )

    model_state_dict = model.state_dict()

    state_dict = {"model": model_state_dict}
    dist_cp_save(state_dict=state_dict, checkpoint_id=checkpoint_path)
    dist.barrier()

    if dist.get_rank() == 0:
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def train(args, model, tokenizer, train_loader, test_loader, optimizer, scheduler, device,  model_type: str = 't5_gemma_2'):
    best_score = float('inf')
    epochs_since_improvement = 0
    eval_every = 2  
    
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    rank = dist.get_rank()
    is_main = rank == 0

    if is_main:
        mlflow.set_tracking_uri(args.address)
        mlflow.set_experiment(args.experiment_name) 

    if is_main:
        run_context = mlflow.start_run(run_name=args.experiment_name)
    else:
        run_context = nullcontext()
    
    
    with run_context as run:
        if is_main:
            mlflow.log_params({
                "model_type": model_type,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "max_n_epochs": args.max_n_epochs,
                "patience_epochs": args.patience_epochs,
                "optimizer_type": args.optimizer_type,
                "experiment_name": args.experiment_name,
            })
        
        for epoch in range(1, args.max_n_epochs + 1):
            train_loader.sampler.set_epoch(epoch)
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
            )
            
            if is_main:
                print(f"Epoch {epoch}: Average train loss was {train_loss}")
                mlflow.log_metric("train_loss", train_loss, step=epoch)
    
            if epoch % eval_every == 0 or epoch == args.max_n_epochs:
                eval_loss= eval_epoch(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                )
                
                if is_main:
                    print(f"Epoch {epoch}: Average eval loss was {eval_loss}")
                    mlflow.log_metric("eval_loss", eval_loss, step=epoch)
        
                if eval_loss < best_score:
                    best_score = eval_loss
                    epochs_since_improvement = 0
                    save_model(checkpoint_dir, model, tokenizer, best=True)
                else:
                    epochs_since_improvement += eval_every
            
            save_model(checkpoint_dir, model, tokenizer, best=False)
            
            should_stop = torch.tensor(
                int(epochs_since_improvement >= args.patience_epochs),
                dtype=torch.int32,
                device=device,
            )
            dist.broadcast(should_stop, src=0)

            if should_stop.item():
                if is_main:
                    print(f"Stopping after {epoch} epochs")
                break

        
        torch.distributed.barrier()
        save_policy = ShardedStateDictConfig(offload_to_cpu=False)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT, save_policy)
        
        model_state_dict = model.state_dict()
        
        state_dict = {"model": model_state_dict}
        
        dist_cp.save(
            state_dict=state_dict,
            checkpoint_id=args.output_dir,
        )
        
        if is_main:
            mlflow.log_metric("final_best", best_score)

            print(
                f"✅ MLflow run finished. "
                f"Run ID: {run.info.run_id}"
            )
            print("🔗 View in UI: mlflow ui")




if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    

    args = SimpleNamespace(
        model_id="google/t5gemma-2-1b-1b",
        batch_size=8,
        checkpoint_dir='checkpoints',
        address="http://0.0.0.0:1234",    
        experiment_name = "dpo_t5gemma2_detox_ru_v1",
        weight_decay=0.05,
        optimizer_type='AdamW',
        scheduler_type='cosine',
        max_n_epochs=10,
        num_warmup_epochs=1,
        patience_epochs=4,
        learning_rate=5e-6,
        device='cuda',    
    )
    
    model_path = "checkpoints/t5_gemma_2_experiments/t5gemma_2_detox_v1/model_best.pt"
    model, tokenizer = load_model_from_checkpoint(
        args,
        model_path,
    )

    train_dataset = load_dataset("r1char9/toxic-dataset-dpo", split='train')
    test_dataset = load_dataset("r1char9/toxic-dataset-dpo", split='test')

    train_tokenized_dataset = train_dataset.map(
        orpo_preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_tokenized_dataset = test_dataset.map(
        orpo_preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    train_sampler = DistributedSampler(
        train_tokenized_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
    )
    train_loader = DataLoader(
        train_tokenized_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=orpo_collate_fn,
        drop_last=True
    )

    test_sampler = DistributedSampler(
        test_tokenized_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_tokenized_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        collate_fn=orpo_collate_fn,
        drop_last=True
    )
    
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Gemma2EncoderLayer,
            T5Gemma2DecoderLayer,
        },
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        mixed_precision=mixed_precision,
        use_orig_params=True,
    )
    
    optimizer, scheduler = (
        initialize_optimizer_and_scheduler(
            args,
            model,
            len(train_loader),
        )
    )

    train(
        args,
        model,
        tokenizer,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
    )

    dist.barrier()
    dist.destroy_process_group()