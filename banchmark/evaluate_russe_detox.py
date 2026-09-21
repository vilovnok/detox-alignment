"""Оценка модели по бенчмарку RUSSE-2022 / ruDetox (STA, SIM, FL, J).

Запуск:
    python evaluate_russe_detox.py
"""

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from russe_detox_metrics import compute_russe_detox_metrics, load_russe_detox_classifiers

MODEL_ID = "r1char9/t5gemma2-detox-ru"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "Детоксифицируй: "

EVAL_DATASET_ID = "s-nlp/ru_paradetox"
TOXIC_COLUMN = "ru_toxic_comment"
LIMIT = None  # None — прогнать весь датасет


@torch.no_grad()
def generate_detox(texts, model, tokenizer, batch_size=16):
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch = [PROMPT + t for t in texts[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        gen = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
        )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

    sim_model = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)
    sta_model, sta_tokenizer, fl_model, fl_tokenizer = load_russe_detox_classifiers(device=DEVICE)

    dataset = load_dataset(EVAL_DATASET_ID, split="train")
    if LIMIT:
        dataset = dataset.select(range(min(LIMIT, len(dataset))))

    originals = dataset[TOXIC_COLUMN]
    detoxed = generate_detox(originals, model, tokenizer)

    metrics = compute_russe_detox_metrics(
        originals,
        detoxed,
        sta_model,
        sta_tokenizer,
        fl_model,
        fl_tokenizer,
        sim_model,
        device=DEVICE,
    )

    print(f"STA: {metrics['STA']:.4f}")
    print(f"SIM: {metrics['SIM']:.4f}")
    print(f"FL:  {metrics['FL']:.4f}")
    print(f"J:   {metrics['J']:.4f}")


if __name__ == "__main__":
    main()