"""RUSSE-2022 / ruDetox benchmark: STA, SIM, FL, J.

Использует те же предобученные классификаторы, что и официальный пайплайн
оценки RUSSE-2022 (https://github.com/s-nlp/russe_detox_2022):

  - STA (Style Transfer Accuracy):  s-nlp/russian_toxicity_classifier
        id2label = {0: "neutral", 1: "toxic"} -> берём P(neutral)
  - FL  (Fluency):                 s-nlp/rubert-base-corruption-detector
        id2label = {0: "unnatural", 1: "natural"} -> берём P(natural)
  - SIM (Meaning Preservation):     LaBSE cosine similarity (вход vs выход)
  - J   (Joint score) = mean(STA * SIM * FL) — сводная метрика бенчмарка
"""

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STA_MODEL_ID = "s-nlp/russian_toxicity_classifier"
FL_MODEL_ID = "s-nlp/rubert-base-corruption-detector"


def load_russe_detox_classifiers(device="cuda"):
    """Загружает STA и FL классификаторы из официального пайплайна RUSSE-2022."""
    sta_tokenizer = AutoTokenizer.from_pretrained(STA_MODEL_ID)
    sta_model = AutoModelForSequenceClassification.from_pretrained(STA_MODEL_ID).to(device).eval()

    fl_tokenizer = AutoTokenizer.from_pretrained(FL_MODEL_ID)
    fl_model = AutoModelForSequenceClassification.from_pretrained(FL_MODEL_ID).to(device).eval()

    return sta_model, sta_tokenizer, fl_model, fl_tokenizer


@torch.no_grad()
def style_transfer_accuracy(texts, sta_model, sta_tokenizer, device="cuda", batch_size=32):
    """STA: средняя уверенность классификатора, что текст НЕ токсичен (label 0 = neutral)."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = sta_tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
        probs = F.softmax(sta_model(**inputs).logits, dim=-1)
        scores.extend(probs[:, 0].cpu().tolist())  # 0 = "neutral"
    return scores


@torch.no_grad()
def fluency_score(texts, fl_model, fl_tokenizer, device="cuda", batch_size=32):
    """FL: средняя уверенность классификатора, что текст естественный (label 1 = natural)."""
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = fl_tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(device)
        probs = F.softmax(fl_model(**inputs).logits, dim=-1)
        scores.extend(probs[:, 1].cpu().tolist())  # 1 = "natural"
    return scores


def meaning_preservation(originals, detoxed, sim_model, batch_size=32):
    """SIM: косинусная близость LaBSE-эмбеддингов исходного и детокс-текста."""
    orig_emb = sim_model.encode(
        originals, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size
    )
    detox_emb = sim_model.encode(
        detoxed, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size
    )
    return util.pairwise_cos_sim(orig_emb, detox_emb).cpu().tolist()


def compute_russe_detox_metrics(
    originals,
    detoxed,
    sta_model,
    sta_tokenizer,
    fl_model,
    fl_tokenizer,
    sim_model,
    device="cuda",
):
    """Считает STA, SIM, FL и J (joint score) по методике RUSSE-2022 / ruDetox.

    originals — исходные токсичные тексты (без промпта "Детоксифицируй: ").
    detoxed   — сгенерированные моделью детокс-версии.
    """
    sta = style_transfer_accuracy(detoxed, sta_model, sta_tokenizer, device)
    fl = fluency_score(detoxed, fl_model, fl_tokenizer, device)
    sim = meaning_preservation(originals, detoxed, sim_model)

    per_sample_j = [s * m * f for s, m, f in zip(sta, sim, fl)]

    return {
        "STA": float(np.mean(sta)),
        "SIM": float(np.mean(sim)),
        "FL": float(np.mean(fl)),
        "J": float(np.mean(per_sample_j)),
        "per_sample": {"sta": sta, "sim": sim, "fl": fl, "j": per_sample_j},
    }