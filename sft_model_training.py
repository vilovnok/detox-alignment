import os
from types import SimpleNamespace

import mlflow
import numpy as np
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from config import ALLOWED_KEYS, LANG_PROMPTS, TENSOR_KEYS
from metrics import compute_metrics
from utils import initialize_model, initialize_optimizer, initialize_scheduler, save_model




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


def make_collate_fns(base_collator):
    """Фабрика: коллаторам нужен base_collator, а DataLoader вызывает их без аргументов."""
 
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
 
    return train_collate_fn, test_collate_fn
 
 
def make_preprocess_fns(tokenizer):
    """Фабрика: препроцессингу нужен tokenizer, а datasets.map() передаёт только examples."""
 
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
 
    return train_preprocess_function, test_preprocess_function
 
 
def main():
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
 
    model, tokenizer = initialize_model(args)
    sim = SentenceTransformer('sentence-transformers/LaBSE')
 
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=args.model_id,
        padding=True,
    )
    train_collate_fn, test_collate_fn = make_collate_fns(base_collator)
    train_preprocess_function, test_preprocess_function = make_preprocess_fns(tokenizer)
 
    train_dataset = load_dataset("r1char9/toxic-detox-pairs", split='train')
    test_dataset = load_dataset("r1char9/toxic-detox-pairs", split='test')
 
    train_dataset = train_dataset.filter(lambda x: x['lang'] in ['ru', 'en'])
    test_dataset = test_dataset.filter(lambda x: x['lang'] in ['ru', 'en'])
 
    test_dataset_small = test_dataset.select(range(1000))
    test_dataset_to_train = test_dataset.select(range(1000, len(test_dataset)))
 
    train_dataset = concatenate_datasets([train_dataset, test_dataset_to_train])
    test_dataset = test_dataset_small
 
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