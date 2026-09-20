import json
import random
import time
from pathlib import Path
from datetime import datetime

import mlflow
import pandas as pd
from tqdm.auto import tqdm

import typer
from openai import OpenAI
from pydantic import BaseModel, Field



BAD_DETOX_TYPES = {
    "residual_toxicity": {
        "name": "Остаточная токсичность",
        "description": (
            "Часть грубой лексики или агрессивной интонации сохранена или заменена "
            "на смягчённый, но всё ещё уничижительный/пассивно-агрессивный аналог "
            "(например, мат заменён на грубое разговорное слово, а не нейтральное)."
        ),
        "instruction": (
            "Перепиши [detoxic] так, чтобы 1-2 грубых или уничижительных слова/оборота "
            "из [toxic] остались в тексте либо в смягчённой, но всё ещё разговорно-агрессивной "
            "форме. Остальной текст оставь близким к [detoxic]."
        ),
    },
    "meaning_loss": {
        "name": "Потеря смысла / деталей",
        "description": (
            "Из текста пропадают важные фактические детали исходной жалобы "
            "(что именно сломано, сроки, требование) — сообщение становится "
            "расплывчатым и малополезным."
        ),
        "instruction": (
            "Сократи [detoxic] так, чтобы убрать конкретные детали из [toxic] "
            "(что именно сломалось, сроки, конкретное требование), оставив только "
            "общую жалобу без подробностей. Результат должен быть короче исходного [detoxic]."
        ),
    },
    "over_softening": {
        "name": "Излишнее смягчение / потеря эмоционального посыла",
        "description": (
            "Текст становится настолько нейтральным и обтекаемым, что теряется "
            "законное недовольство автора — звучит равнодушно или как будто "
            "ничего плохого не произошло."
        ),
        "instruction": (
            "Перепиши [detoxic] в подчёркнуто нейтральном, почти безразличном тоне, "
            "убрав любые следы недовольства или срочности, присутствовавшие в [toxic]. "
            "Текст не должен звучать как жалоба."
        ),
    },
    "register_mismatch": {
        "name": "Несоответствие регистра (шаблонность)",
        "description": (
            "Ответ превращается в общий канцелярский шаблон, никак не связанный "
            "с конкретной ситуацией из [toxic] — как будто сгенерирован под любую жалобу."
        ),
        "instruction": (
            "Перепиши сообщение как максимально общий канцелярский шаблон обращения "
            "в поддержку, без единой конкретной детали из [toxic] (не упоминай товар, "
            "поломку, сроки — только общие фразы вида 'прошу рассмотреть обращение')."
        ),
    },
    "hallucination": {
        "name": "Добавление несуществующих деталей",
        "description": (
            "В текст добавляются факты, детали или требования, которых не было "
            "ни в [toxic], ни в [detoxic]."
        ),
        "instruction": (
            "Возьми [detoxic] и добавь 1-2 придуманные детали, которых нет ни в [toxic], "
            "ни в [detoxic] (например, номер заказа, дату, дополнительное требование "
            "вроде компенсации), сделав текст фактически недостоверным."
        ),
    },
    "incomplete_rewrite": {
        "name": "Неполная переработка",
        "description": (
            "Детоксифицирована только часть сообщения, а другая часть почти "
            "дословно повторяет токсичный оригинал."
        ),
        "instruction": (
            "Составь текст, в котором первая половина — это переработанная нейтральная "
            "версия по смыслу [detoxic], а вторая половина почти дословно повторяет "
            "агрессивные формулировки из [toxic] (без мата, но с сохранением грубого тона)."
        ),
    },
    "surface_masking": {
        "name": "Поверхностная маскировка (эвфемизмы/цензура)",
        "description": (
            "Мат просто заменён на звёздочки/эвфемизмы ('бл*, бл*н') без реальной "
            "перефразировки предложения — структура и тон остаются токсичными."
        ),
        "instruction": (
            "Возьми [toxic] и просто замени в нём матерные и оскорбительные слова "
            "на аналоги со звёздочками или эвфемизмы (например 'долбоёб' → 'д**б'), "
            "не меняя структуру предложения, синтаксис и порядок слов."
        ),
    },
    "wrong_tone_shift": {
        "name": "Смена тона в неверную сторону",
        "description": (
            "Токсичность убрана, но текст стал издевательски-саркастичным или "
            "пассивно-агрессивным, а не нейтрально-деловым."
        ),
        "instruction": (
            "Перепиши [detoxic] в саркастично-пассивно-агрессивном тоне: "
            "формально вежливые слова, но с явным сарказмом или подколками, "
            "вместо искренне нейтрального делового тона."
        ),
    },
    "grammar_broken": {
        "name": "Грамматическая/синтаксическая поломка",
        "description": (
            "После переработки текст содержит явные грамматические ошибки, "
            "рассогласования или оборванные конструкции."
        ),
        "instruction": (
            "Перепиши [detoxic], намеренно внеся 1-2 явные грамматические или "
            "синтаксические ошибки (рассогласование падежей/чисел, оборванное "
            "предложение), сохранив нейтральность тона."
        ),
    },
    "exact_copy": {
        "name": "Точная копия токсичного текста",
        "description": (
            "Токсичный текст вообще не изменён — просто продублирован. "
            "Самый слабый и дешёвый негативный пример, использовать редко."
        ),
        "instruction": (
            "Верни [toxic] практически без изменений, максимум убрав одно матерное слово."
        ),
    },
}


SYSTEM_PROMPT = """
Ты помогаешь собрать датасет для обучения модели детоксификации текста методом DPO (Direct Preference Optimization).
 
Твоя задача — генерировать НЕУДАЧНЫЕ примеры детоксификации: текст, который выглядит
как попытка убрать токсичность из исходного токсичного текста, но содержит конкретный
недостаток. Тип недостатка и инструкция по его воспроизведению будут указаны в запросе
пользователя вместе с исходным токсичным текстом и эталонным примером качественной
детоксификации.
 
Общие требования к результату:
- Текст должен быть на русском языке, одним связным текстом.
- Не добавляй пояснения, кавычки, заголовки или комментарии о том, что текст плохой —
  выведи только сам текст детоксификации.
- Строго следуй указанному типу недостатка и инструкции по его воспроизведению.
 
Формат ответа:
Верни ТОЛЬКО валидный JSON, соответствующий предоставленной схеме (поле bad_detoxified_text).
Не включай разметку Markdown, комментарии, пояснения или любой текст вне JSON.
""".strip()
 
BAD_DETOX_PROMPT_TEMPLATE = """
[toxic]
{toxic}
 
[detoxic] (эталонный качественный пример детоксификации, для ориентира по смыслу и объёму)
{detoxic}
 
Тип недостатка, который нужно воспроизвести: {failure_name}
Описание типа: {failure_description}
 
Инструкция по генерации:
{failure_instruction}
""".strip()

app = typer.Typer()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty",
)

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
MAX_TOKENS = 6096


class DetoxificationSample(BaseModel):
    bad_detoxified_text: str = Field(
        description=(
            "Russian rewrite of the toxic text that attempts to preserve "
            "the original meaning and communicative intent, but provides an "
            "inferior detoxification compared with the preferred rewrite. "
            "The text may retain mild profanity, insults, vulgar or overly "
            "aggressive wording, or use unnatural, awkward, overly emotional, "
            "or only partially neutralized language."
        )
    )



def sample_bad_detox_prompt(toxic: str, detoxic: str,
                             weights: dict[str, float] | None = None,
                             rng: random.Random | None = None) -> tuple[str, str]:
    """
    Сэмплирует тип "плохой" детоксификации и возвращает
    (заполненный промпт для генератора, ключ выбранного типа).
    """
    rng = rng or random
    keys = list(BAD_DETOX_TYPES.keys())

    if weights is None:
        default_weights = {
            "residual_toxicity": 1.5,
            "meaning_loss": 1.5,
            "over_softening": 1.2,
            "register_mismatch": 1.0,
            "hallucination": 1.0,
            "incomplete_rewrite": 1.2,
            "surface_masking": 0.7,
            "wrong_tone_shift": 1.0,
            "grammar_broken": 0.6,
            "exact_copy": 0.3,
        }
        weights = default_weights

    w = [weights.get(k, 1.0) for k in keys]
    chosen_key = rng.choices(keys, weights=w, k=1)[0]
    spec = BAD_DETOX_TYPES[chosen_key]

    prompt = BAD_DETOX_PROMPT_TEMPLATE.format(
        toxic=toxic,
        detoxic=detoxic,
        failure_name=spec["name"],
        failure_description=spec["description"],
        failure_instruction=spec["instruction"],
    )
    return prompt, chosen_key


def generate_sample(user_prompt: str) -> DetoxificationSample:
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.9,
        max_tokens=MAX_TOKENS,
        response_format=DetoxificationSample,
    )

    return response.choices[0].message.parsed


def generate_dataset_sample(toxic: str, detoxic: str, rng: random.Random) -> dict:
    gen_prompt, ftype = sample_bad_detox_prompt(toxic, detoxic, rng=rng)
    rejected_sample = generate_sample(gen_prompt)
 
    output_dict = {
        "toxic_text": toxic,
        "chosen": detoxic,
        "rejected": rejected_sample.bad_detoxified_text,
        "rejected_type": ftype,
    }
    
    return output_dict



def generate_dataset(
    df: pd.DataFrame,
    start: int,
    end: int,
    experiment_name: str = "russian-text-detoxification-dpo",
    run_name: str = "qwen3.6-35b-a3b-synthetic-dataset",
    log_every: int = 50,
    save_every: int = 500,
    seed: int = 42,
):
    """
    Генерирует DPO-сэмплы только для среза df[start:end].
    Используется для шардирования обработки между несколькими
    параллельными процессами (например, start=0/end=2500,
    start=2500/end=5000 и т.д.), не пересекающимися по данным.
    """
    total_rows = len(df)
 
    if start < 0:
        raise ValueError(f"start не может быть отрицательным: {start}")
    if end > total_rows:
        raise ValueError(
            f"end={end} превышает размер датафрейма ({total_rows} строк)"
        )
    if start >= end:
        raise ValueError(f"start ({start}) должен быть меньше end ({end})")
 
    shard_df = df.iloc[start:end]
    n_samples = len(shard_df)
 
    mlflow.set_tracking_uri("http://localhost:1234")
    mlflow.set_experiment(experiment_name)
 
    output_dir = Path("detox_dataset_dpo")
    output_dir.mkdir(parents=True, exist_ok=True)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_filename = f"samples_{start}_{end}_{timestamp}.jsonl"
    samples_path = output_dir / samples_filename
 
    shard_run_name = f"{run_name}_{start}_{end}"
 
    generated = 0
    failed = 0
    start_time = time.time()
 
    rng = random.Random(seed)
 
    with mlflow.start_run(run_name=shard_run_name):
        mlflow.log_params(
            {
                "model": MODEL,
                "n_samples": n_samples,
                "shard_start": start,
                "shard_end": end,
                "temperature": 0.9,
                "max_tokens": MAX_TOKENS,
                "seed": seed,
            }
        )
 
        with open(samples_path, "w", encoding="utf-8") as f:
            pbar = tqdm(
                shard_df.itertuples(index=False),
                total=n_samples,
                desc=f"Generating dataset [{start}:{end}]",
            )
 
            for i, row in enumerate(pbar):
                toxic = row.toxic_text
                detoxic = row.detoxified_text
 
                try:
                    sample = generate_dataset_sample(toxic, detoxic, rng)
 
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    f.flush()
 
                    generated += 1
 
                except Exception as e:
                    failed += 1
                    print(f"\n[ERROR] Sample {start + i}: {type(e).__name__}: {e}")
 
                if (i + 1) % log_every == 0:
                    elapsed = time.time() - start_time
                    processed = generated + failed
 
                    success_rate = generated / processed if processed > 0 else 0.0
                    samples_per_sec = processed / elapsed if elapsed > 0 else 0.0
                    avg_time = elapsed / processed if processed > 0 else 0.0
 
                    mlflow.log_metrics(
                        {
                            "generated": generated,
                            "failed": failed,
                            "success_rate": success_rate,
                            "samples_per_second": samples_per_sec,
                            "avg_sample_time_sec": avg_time,
                        },
                        step=i + 1,
                    )
 
                    pbar.set_postfix(
                        {
                            "generated": generated,
                            "failed": failed,
                            "success": f"{success_rate:.2%}",
                            "samples/s": f"{samples_per_sec:.2f}",
                        }
                    )
 
                if (i + 1) % save_every == 0:
                    mlflow.log_artifact(str(samples_path), artifact_path="checkpoints")
 
        elapsed = time.time() - start_time
        success_rate = generated / n_samples if n_samples > 0 else 0.0
        samples_per_sec = generated / elapsed if elapsed > 0 else 0.0
 
        mlflow.log_metrics(
            {
                "generated": generated,
                "failed": failed,
                "success_rate": success_rate,
                "total_time_sec": elapsed,
                "samples_per_second": samples_per_sec,
            },
            step=n_samples,
        )
 
        mlflow.log_artifact(str(samples_path), artifact_path="dataset")
 
        print("\nGeneration finished.")
        print(f"Shard:     [{start}:{end}]")
        print(f"Generated: {generated}")
        print(f"Failed:    {failed}")
        print(f"Time:      {elapsed / 60:.2f} min")
        print(f"Speed:     {samples_per_sec:.2f} samples/sec")
 
    return samples_path



@app.command()
def main(
    csv_path: str = typer.Option(
        "detox_dataset/detox_combined_final_part.csv",
        "--csv-path",
        help="Путь к исходному CSV с колонками toxic_text/detoxified_text",
    ),
    start: int = typer.Option(
        ...,
        "--start",
        help="Индекс первой строки датафрейма (включительно), с которой начинать обработку",
    ),
    end: int = typer.Option(
        ...,
        "--end",
        help="Индекс последней строки датафрейма (не включительно), на которой закончить обработку",
    ),
    experiment_name: str = typer.Option(
        "russian-text-detoxification-dpo",
        "--experiment-name",
        help="Имя эксперимента в MLflow",
    ),
    run_name: str = typer.Option(
        "qwen3.6-35b-a3b-synthetic-dataset",
        "--run-name",
        help="Базовое имя run'а в MLflow (к нему добавится _{start}_{end})",
    ),
    log_every: int = typer.Option(50, "--log-every", help="Как часто логировать метрики"),
    save_every: int = typer.Option(
        500, "--save-every", help="Как часто сохранять промежуточный чекпоинт в MLflow"
    ),
    seed: int = typer.Option(42, "--seed", help="Seed для воспроизводимости выбора типа rejected"),
):
    """
    Пример запуска для декомпозиции обработки между процессами:
 
        python generate_dpo_dataset.py --start 0 --end 2500
        python generate_dpo_dataset.py --start 2500 --end 5000
        python generate_dpo_dataset.py --start 5000 --end 7500
 
    Диапазоны [start:end) не пересекаются, каждый процесс пишет
    в свой собственный файл samples_{start}_{end}_{timestamp}.jsonl.
    """
    df = pd.read_csv(csv_path)
 
    samples_path = generate_dataset(
        df,
        start=start,
        end=end,
        experiment_name=experiment_name,
        run_name=run_name,
        log_every=log_every,
        save_every=save_every,
        seed=seed,
    )
    print(samples_path)


if __name__ == "__main__":
    app()
