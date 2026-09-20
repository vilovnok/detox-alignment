import json
import random
import time
from pathlib import Path
from datetime import datetime
import random

import mlflow
import pandas as pd
from tqdm.auto import tqdm

from openai import OpenAI
from pydantic import BaseModel, Field



client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty",
)

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


class DetoxificationSample(BaseModel):
    toxic_text: str = Field(
        description=(
            "Original Russian text containing profanity, insults, "
            "vulgarity, rude or aggressive language."
        )
    )
    detoxified_text: str = Field(
        description=(
            "Natural Russian rewrite that preserves the meaning and "
            "communicative intent while removing profanity, insults, "
            "vulgarity and unnecessarily aggressive wording."
        )
    )


SYSTEM_PROMPT = """\
Вы — генератор синтетических наборов данных для обучения модели детоксикации текста.

Ваша задача — генерировать реалистичные примеры пар токсичный–детоксицированный текст на русском языке.

Каждый пример должен содержать:

1. `toxic_text` — исходный токсичный, грубый, оскорбительный, вульгарный или содержащий ненормативную лексику текст.
2. `detoxified_text` — переписанную версию того же текста, которая сохраняет исходный смысл и коммуникативное намерение, но удаляет ненормативную лексику, оскорбления, вульгарные выражения и излишне агрессивные формулировки.

Детоксицированный текст должен сохранять:

* основной смысл исходного текста;
* намерение говорящего;
* важные факты и детали;
* адресата высказывания;
* исходную просьбу, жалобу, критику или мнение;
* примерно тот же объём информации.

Детоксицированный текст НЕ должен:

* содержать ненормативную или бранную лексику;
* содержать прямые оскорбления;
* заменять нецензурную лексику частично замаскированными вариантами, такими как «***», «бл***», «сука» и т.п.;
* вводить новую информацию;
* изменять смысл оригинала;
* превращать жалобу в бессвязное утверждение;
* становиться излишне формальным или неестественным;
* просто удалять проблемное предложение, если его смысл можно сохранить.

Детоксикация должна звучать так, как естественно написал бы обычный русскоговорящий человек.

Важно:

* Токсичный текст должен быть реалистичным и разнообразным.
* Токсичные примеры могут содержать мягкую или сильную нецензурную лексику, оскорбления, грубые выражения, агрессивные жалобы, враждебные формулировки или их комбинации.
* Не делайте все примеры чрезмерно оскорбительными. Включайте разные уровни токсичности.
* Токсичный текст может состоять из одного предложения или из нескольких предложений.
* Детоксицированный текст также может состоять из одного или нескольких предложений.
* По возможности сохраняйте структуру предложений, но приоритетом является сохранение смысла.
* Не делайте токсичную версию искусственно абсурдной только ради того, чтобы вставить ненормативную лексику.
* Токсичная и детоксицированная версии должны описывать одну и ту же ситуацию.
* Детоксицированная версия должна по-прежнему выражать критику, недовольство, несогласие или негативные эмоции, если они присутствуют в оригинале. Не делайте её автоматически позитивной.
* Не добавляйте извинения, формулы вежливости или пояснения, если они не необходимы для естественной детоксикации.

Примеры преобразований:

Токсичный:
«Этот идиот опять прислал мне какую-то хрень, невозможно с этим работать.»

Детоксицированный:
«Этот человек опять прислал мне непонятный материал, с которым невозможно работать.»

Токсичный:
«Да пошли вы нахуй, ублюдки, со своими правилами, заебали уже каждый раз одно и то же требовать.»

Детоксицированный:
«Не согласен с вашими правилами, уже надоело каждый раз выполнять одно и то же требование.»

Токсичный:
«Ты что, совсем еблан? Я же уже объяснил это три раза.»

Детоксицированный:
«С тобой всё хорошо? Я уже объяснил это три раза.»

Приведённые выше примеры являются лишь иллюстративными. Создавайте новые ситуации и не копируйте их.

Для каждого сгенерированного примера независимо проверьте, что:

1. `toxic_text` действительно содержит токсичную, грубую, оскорбительную, вульгарную или нецензурную лексику.
2. `detoxified_text` не содержит ненормативной лексики и прямых оскорблений.
3. Оба текста относятся к одной и той же ситуации.
4. Основное смысловое содержание сохранено.
5. Детоксицированный текст является естественным русским языком.
6. Никакая важная информация не была добавлена или удалена.

Верните ТОЛЬКО валидный JSON, соответствующий предоставленной схеме.
Не включайте разметку Markdown, комментарии, пояснения или дополнительный текст вне JSON.
""".strip()

SCENARIOS = [
    "рабочая переписка",
    "конфликт с коллегой",
    "спор с руководителем",
    "обсуждение задачи в команде",
    "обсуждение сроков проекта",
    "критика работы коллеги",
    "ошибка сотрудника",
    "разбор проваленной задачи",
    "конфликт из-за распределения обязанностей",
    "обсуждение зарплаты или повышения",
    "переписка после неприятного рабочего созвона",

    "служба поддержки",
    "жалоба на сервис",
    "недовольство доставкой",
    "задержка заказа",
    "неправильный заказ",
    "сломанный или бракованный товар",
    "возврат товара",
    "спор с продавцом",
    "спор о покупке",
    "некачественная услуга",
    "отмена подписки",
    "проблема с оплатой",

    "техническая проблема",
    "ошибка в программе",
    "проблема с интернетом",
    "проблема с компьютером",
    "проблема со смартфоном",
    "обсуждение багов",
    "спор о выборе технологии",
    "обсуждение программного обеспечения",
    "раздражение из-за медленного приложения",

    "комментарий в интернете",
    "спор на форуме",
    "спор в социальных сетях",
    "обсуждение новости",
    "обсуждение политики компании",
    "спор о правилах сообщества",
    "реакция на чужой комментарий",
    "конфликт в игровом чате",

    "обсуждение игры",
    "спор между игроками",
    "жалоба на разработчиков игры",
    "обсуждение фильма",
    "спор о сериале",
    "обсуждение музыки",
    "критика книги",

    "спор о цене товара",
    "обсуждение возврата денег",
    "спор из-за списания денег",
    "недовольство условиями подписки",
    "спор с банком или финансовым сервисом",

    "бытовой конфликт",
    "спор между друзьями",
    "ссора с соседом",
    "конфликт с родственником",
    "спор из-за бытовых обязанностей",
    "конфликт из-за шума",
    "спор из-за опоздания",
    "раздражение из-за нарушения договорённости",

    "конфликт в общественном транспорте",
    "спор в очереди",
    "конфликт в магазине",
    "конфликт в ресторане или кафе",
    "жалоба на поведение другого человека",
    "спор из-за парковки",

    "спор с преподавателем",
    "обсуждение оценки",
    "конфликт с одногруппником",
    "жалоба на учебный процесс",
    "обсуждение группового проекта",
]

TONES = [
    "раздражение",
    "сарказм",
    "грубая претензия",
    "открытая агрессия",
    "насмешка",
    "возмущение",
    "недовольство",
    "разочарование",
    "нетерпение",
]

TEXT_LENGTHS = [
    "одно предложение",
    "2–3 предложения",
    "3–5 предложений",
]

TOXICITY_LEVELS = [
    "мягкая грубость",
    "умеренная токсичность",
    "сильная грубость",
    "выраженная токсичность",
]

FORMATS = [
    "сообщение в чате",
    "комментарий",
    "отзыв",
    "диалоговая реплика",
    "жалоба",
    "ответ на сообщение",
    "письмо",
]



## Исходное использование кода для генерации


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
        max_tokens=4096,
        response_format=DetoxificationSample,
    )

    return response.choices[0].message.parsed



def generate_dataset_sample():
    scenario = random.choice(SCENARIOS)
    tone = random.choice(TONES)
    toxicity = random.choice(TOXICITY_LEVELS)
    format_type = random.choice(FORMATS)
    length = random.choice(TEXT_LENGTHS)


    prompt = f"""
Создай реалистичную ситуацию.

Контекст: {random.choice(SCENARIOS)}
Тон: {random.choice(TONES)}
Уровень токсичности: {random.choice(TOXICITY_LEVELS)}
Формат: {random.choice(FORMATS)}
Количество предложений: {random.choice(TEXT_LENGTHS)}

Токсичный текст должен содержать естественную русскую ненормативную лексику, оскорбления или грубые/агрессивные формулировки. 
Детоксифицированный текст должен сохранять смысл без ненормативной лексики или оскорблений
""".strip()

    sample = generate_sample(prompt)

    output_dict = {
        "scenario": scenario,
        "tone": tone,
        "toxicity_level": toxicity,
        "format": format_type,
        "text_length": length,
        "toxic_text": sample.toxic_text,
        "detoxified_text": sample.detoxified_text
    }
    return output_dict






def generate_dataset(
    n_samples: int = 10_000,
    experiment_name: str = "russian-text-detoxification",
    run_name: str = "qwen3.6-35b-a3b-synthetic-dataset",
    log_every: int = 50,
    save_every: int = 500,
):
    mlflow.set_tracking_uri("http://localhost:1234")
    mlflow.set_experiment(experiment_name)

    output_dir = Path("detox_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_filename = f"samples_{timestamp}.jsonl"
    samples_path = output_dir / samples_filename
    # samples_path = output_dir / "samples.jsonl"

    generated = 0
    failed = 0
    start_time = time.time()
    
    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "model": MODEL,
            "n_samples": n_samples,
            "temperature": 0.9,
            "max_tokens": 6096,
            "n_scenarios": len(SCENARIOS),
            "n_tones": len(TONES),
            "n_toxicity_levels": len(TOXICITY_LEVELS),
            "n_formats": len(FORMATS),
            "n_text_lengths": len(TEXT_LENGTHS),
        })

        with open(samples_path, "w", encoding="utf-8") as f:

            pbar = tqdm(
                range(n_samples),
                total=n_samples,
                desc="Generating dataset",
            )

            for i in pbar:
                sample_start = time.time()

                try:
                    sample = generate_dataset_sample()

                    f.write(
                        json.dumps(
                            sample,
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    f.flush()

                    generated += 1

                except Exception as e:
                    failed += 1

                    print(
                        f"\n[ERROR] Sample {i}: "
                        f"{type(e).__name__}: {e}"
                    )

                if (i + 1) % log_every == 0:

                    elapsed = time.time() - start_time
                    processed = generated + failed

                    success_rate = (
                        generated / processed
                        if processed > 0
                        else 0.0
                    )

                    samples_per_sec = (
                        processed / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    avg_time = (
                        elapsed / processed
                        if processed > 0
                        else 0.0
                    )

                    mlflow.log_metrics({
                        "generated": generated,
                        "failed": failed,
                        "success_rate": success_rate,
                        "samples_per_second": samples_per_sec,
                        "avg_sample_time_sec": avg_time,
                    }, step=i + 1)

                    pbar.set_postfix({
                        "generated": generated,
                        "failed": failed,
                        "success": f"{success_rate:.2%}",
                        "samples/s": f"{samples_per_sec:.2f}",
                    })

                if (i + 1) % save_every == 0:
                    mlflow.log_artifact(
                        str(samples_path),
                        artifact_path="checkpoints",
                    )


        elapsed = time.time() - start_time

        success_rate = (
            generated / n_samples
            if n_samples > 0
            else 0.0
        )

        samples_per_sec = (
            generated / elapsed
            if elapsed > 0
            else 0.0
        )

        mlflow.log_metrics({
            "generated": generated,
            "failed": failed,
            "success_rate": success_rate,
            "total_time_sec": elapsed,
            "samples_per_second": samples_per_sec,
        }, step=n_samples)

        mlflow.log_artifact(
            str(samples_path),
            artifact_path="dataset",
        )

        print("\nGeneration finished.")
        print(f"Generated: {generated}")
        print(f"Failed:    {failed}")
        print(f"Time:      {elapsed / 60:.2f} min")
        print(f"Speed:     {samples_per_sec:.2f} samples/sec")

    return samples_path

if __name__ == "__main__":
    samples_path = generate_dataset(
        n_samples=10_000,
    )
    print(samples_path)