import os
from types import SimpleNamespace

import mlflow
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from metrics import compute_metrics
from utils import mkdir
from config import ALLOWED_KEYS, TENSOR_KEYS, LANG_PROMPTS



def initialize_model(args):
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16
    )

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_count = 0
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Linear)
            and "vision_tower" not in name
            and any(name.endswith(suffix) for suffix in target_suffixes)
        ):
            for param in module.parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"{trainable_params:,}  parameters out of {total_params:,} ")
    return model.to(args.device), tokenizer


def save_model(checkpoint_dir, model, best):
    mkdir(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{'best' if best else 'latest'}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_path)
    return model_path


def load_model_from_checkpoint(args, model_path):
    model, tokenizer = initialize_model(args)
    if not os.path.exists(model_path):
        return model, tokenizer

    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer



def train_epoch(model, train_loader, optimizer, scheduler):
    model.train()

    total_loss = 0
    total_tokens = 0

    for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
        optimizer.zero_grad()
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs['loss']

        non_pad = batch['labels'] != -100

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    return total_loss / total_tokens


def eval_epoch(model, tokenizer, sim, test_loader, epoch=0):
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_mean_sims = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", total=len(test_loader)):
            model_inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
                "labels": batch["labels"].to(model.device),
            }

            outputs = model(**model_inputs)
            loss = outputs['loss']
            targets = batch['labels']

            non_pad = targets != -100
            num_tokens = torch.sum(non_pad).item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            batch_metrics = compute_metrics(
                toxic_texts=batch['detox_comment'],
                neutral_texts=batch['neutral_comment'],
                model=model,
                tokenizer=tokenizer,
                sim_model=sim,
                epoch=epoch,
            )
            all_mean_sims.extend(batch_metrics["per_sample_sim"])

    eval_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    mean_sim = float(np.mean(all_mean_sims)) if all_mean_sims else 0.0
    return eval_loss, mean_sim


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    elif args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9
        )
    else:
        raise ValueError(f"misisng: {args.optimizer_type}")

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def train(args, model, tokenizer, sim, train_loader, test_loader, optimizer, scheduler, model_type: str = 't5_gemma_2'):
    best_score = -1
    epochs_since_improvement = 0
    eval_every = 2

    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    mlflow.set_tracking_uri(args.address)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.experiment_name) as run:
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
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            print(f"Epoch {epoch}: Average train loss was {train_loss}")
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if epoch % eval_every == 0 or epoch == args.max_n_epochs:
                eval_loss, mean_sim = eval_epoch(
                    model, tokenizer, sim, test_loader, epoch=epoch
                )
                print(f"Epoch {epoch}: Test loss {eval_loss} | score: {mean_sim}")
                mlflow.log_metrics({"eval_loss": eval_loss, "mean_similarity": mean_sim}, step=epoch)

                if mean_sim > best_score:
                    best_score = mean_sim
                    epochs_since_improvement = 0
                    save_model(checkpoint_dir, model, best=True)
                    mlflow.log_metric("best_score", best_score, step=epoch)
                else:
                    epochs_since_improvement += eval_every

            save_model(checkpoint_dir, model, best=False)

            if epochs_since_improvement >= args.patience_epochs:
                print(f" stopping after {epoch} epochs ")
                break

        mlflow.log_metric("final_best", best_score)
        print(f"✅ MLflow run finished. Run ID: {run.info.run_id}")
        print(f"🔗 View in UI: mlflow ui")




args = SimpleNamespace(
    model_id="google/t5gemma-2-1b-1b",
    batch_size=16,
    checkpoint_dir='checkpoints',
    address="http://0.0.0.0:1234",
    experiment_name="sft_t5gemma2_detox_ru_v1",
    weight_decay=0.05,
    optimizer_type='AdamW',
    scheduler_type='cosine',
    max_n_epochs=10,
    num_warmup_epochs=1,
    patience_epochs=4,
    learning_rate=5e-5,
    device='cuda',
)

def train_collate_fn(features):
    filtered = [
        {k: v for k, v in f.items() if k in ALLOWED_KEYS}
        for f in features
    ]
    return base_collator(filtered)

def test_collate_fn(features):
    tensor_features = [
        {k: v for k, v in f.items() if k in TENSOR_KEYS}
        for f in features
    ]
    batch = base_collator(tensor_features)

    text_keys = set(features[0].keys()) - TENSOR_KEYS
    for key in text_keys:
        batch[key] = [f[key] for f in features]

    return batch


def train_preprocess_function(examples):
    toxic = [LANG_PROMPTS[lang] + tox for lang, tox in zip(examples['lang'], examples['toxic_comment'])]
    inputs = tokenizer(toxic, truncation=True, max_length=512, add_special_tokens=True)

    targets = [t + tokenizer.eos_token for t in examples['neutral_comment']]
    labels = tokenizer(targets, truncation=True, max_length=512, add_special_tokens=False)

    return {**inputs, 'labels': labels.input_ids}

def test_preprocess_function(examples):
    toxic = [LANG_PROMPTS[lang] + tox for lang, tox in zip(examples['lang'], examples['toxic_comment'])]
    inputs = tokenizer(toxic, truncation=True, max_length=512, add_special_tokens=True)

    targets = [t + tokenizer.eos_token for t in examples['neutral_comment']]
    labels = tokenizer(targets, truncation=True, max_length=512, add_special_tokens=False)

    return {**inputs, 'labels': labels.input_ids, 'detox_comment': toxic, 'neutral_comment': targets}



def main():
    model, tokenizer = initialize_model(args)
    sim = SentenceTransformer('sentence-transformers/LaBSE')

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=args.model_id,
        padding=True,
    )

    train_dataset = load_dataset("r1char9/toxic-detox-pairs", split='train')
    test_dataset = load_dataset("r1char9/toxic-detox-pairs", split='test')
 
    train_dataset = train_dataset.filter(lambda x: x['lang'] in ['ru', 'en'])
    test_dataset = test_dataset.filter(lambda x: x['lang'] in ['ru', 'en'])
 
    test_dataset_small = test_dataset.select(range(1000))
    test_dataset_to_train = test_dataset.select(range(1000, len(test_dataset)))
 
    train_dataset = concatenate_datasets([train_dataset, test_dataset_to_train])
    test_dataset = test_dataset_small
 
    train_preprocess_function = train_preprocess_function(tokenizer)
    test_preprocess_function = test_preprocess_function(tokenizer)
 
    train_tokenized_dataset = train_dataset.map(
        train_preprocess_function, batched=True, remove_columns=train_dataset.column_names
    )
    test_tokenized_dataset = test_dataset.map(
        test_preprocess_function, batched=True, remove_columns=test_dataset.column_names
    )

    train_loader = DataLoader(
        train_tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn
    )
    test_loader = DataLoader(
        test_tokenized_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_collate_fn
    )

    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    train(args, model, tokenizer, sim, train_loader, test_loader, optimizer, scheduler)


if __name__ == "__main__":
    main()