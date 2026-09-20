import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import util


def detoxify(text, model, tokenizer):
    encodings = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            **encodings.to(model.device),
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            num_beam_groups=5,
            diversity_penalty=2.5,
            num_return_sequences=5,
            early_stopping=True,
            trust_remote_code=True,
            do_sample=False,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def compute_metrics(toxic_texts, neutral_texts, model, tokenizer, sim_model, epoch):
    all_scores = []

    for toxic_text, neutral_text in tqdm(
        zip(toxic_texts, neutral_texts),
        desc=f"🔄 Epoch {epoch+1} detoxing",
        unit="seq",
        total=len(toxic_texts),
    ):
        detoxs = detoxify(toxic_text, model, tokenizer)

        embeddings = sim_model.encode(
            [neutral_text] + detoxs,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        neutral_emb = embeddings[0:1]
        detox_embs = embeddings[1:]

        cos_sims = util.cos_sim(neutral_emb, detox_embs).squeeze(0)
        all_scores.append(cos_sims.mean().item())

    return {
        "mean_sim": np.mean(all_scores),
        "per_sample_sim": all_scores,
        "std_sim": np.std(all_scores),
    }