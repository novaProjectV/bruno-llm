#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple


def clamp_lengths(user_text: str, bruno_text: str, max_total_chars: int = 420) -> Tuple[str, str]:
    total = len(user_text) + len(bruno_text)
    if total <= max_total_chars:
        return user_text, bruno_text

    overflow = total - max_total_chars
    keep = max(60, len(bruno_text) - overflow - 3)
    return user_text, bruno_text[:keep].rstrip() + "..."


def family_explain(rng: random.Random) -> Tuple[str, str]:
    concepts = [
        "REST API",
        "JWT",
        "кеширование",
        "CI/CD",
        "Docker",
        "Git rebase",
        "индексы в SQL",
        "нормализация БД",
        "асинхронность в Python",
        "event loop",
        "unit-тест",
        "интеграционный тест",
        "feature flag",
        "blue-green deployment",
        "idempotency",
        "rate limiting",
        "load balancing",
        "NoSQL",
        "CAP theorem",
        "TLS",
        "OAuth 2.0",
        "webhook",
        "ETL",
        "data pipeline",
    ]
    concept = rng.choice(concepts)
    user = f"Объясни простыми словами, что такое {concept}."
    bruno = (
        f"{concept} — это практический инструмент для надёжной работы системы. "
        "Сначала подумай о задаче, затем о правилах применения и только потом о деталях реализации."
    )
    return clamp_lengths(user, bruno)


def family_compare(rng: random.Random) -> Tuple[str, str]:
    pairs = [
        ("list", "tuple"),
        ("process", "thread"),
        ("SQL", "NoSQL"),
        ("Docker image", "container"),
        ("PUT", "PATCH"),
        ("HTTP", "WebSocket"),
        ("sync", "async"),
        ("monolith", "microservices"),
        ("BFS", "DFS"),
        ("CPU-bound", "IO-bound"),
        ("REST", "GraphQL"),
        ("JWT", "session cookie"),
    ]
    left, right = rng.choice(pairs)
    user = f"В чем разница между {left} и {right}?"
    bruno = (
        f"{left} и {right} решают похожую задачу, но с разными компромиссами. "
        f"Выбирай {left}, когда важна простота, и {right}, когда важны гибкость или производительность в конкретном сценарии."
    )
    return clamp_lengths(user, bruno)


def family_plan(rng: random.Random) -> Tuple[str, str]:
    goals = [
        "подготовиться к собеседованию по backend за 2 недели",
        "запустить MVP за 30 дней",
        "наладить ежедневное изучение английского",
        "поднять дисциплину в учебе",
        "снизить технический долг в команде",
        "улучшить качество кода в проекте",
        "перестать откладывать важные задачи",
        "подготовить продукт к публичному релизу",
        "научиться системному дизайну",
        "перейти из junior в middle",
    ]
    goal = rng.choice(goals)
    user = f"Составь короткий план: как {goal}?"
    bruno = (
        "1) Зафиксируй конкретный результат и метрику. "
        "2) Разбей на недельные этапы и ежедневные задачи. "
        "3) Каждые 3-4 дня делай ревью прогресса и корректируй план."
    )
    return clamp_lengths(user, bruno)


def family_code(rng: random.Random) -> Tuple[str, str]:
    languages = ["Python", "JavaScript", "TypeScript", "Go", "Java", "Rust"]
    tasks = [
        "функция для дедупликации массива",
        "валидация email",
        "чтение JSON-файла и подсчет записей",
        "retry с экспоненциальной задержкой",
        "простой rate limiter",
        "кэш с TTL",
        "парсер аргументов CLI",
        "обработка ошибок при HTTP-запросе",
        "поиск максимальной подстроки",
        "очередь задач в памяти",
    ]
    lang = rng.choice(languages)
    task = rng.choice(tasks)
    user = f"Дай пример на {lang}: {task}."
    bruno = (
        "Сделай сначала минимальную рабочую версию, затем добавь проверки входа и обработку ошибок. "
        "Если нужно, я дам сразу готовый код и короткие тесты."
    )
    return clamp_lengths(user, bruno)


def family_debug(rng: random.Random) -> Tuple[str, str]:
    issues = [
        "API отвечает 500 только в проде",
        "запросы к БД стали медленными после релиза",
        "память процесса растет со временем",
        "веб-сокет периодически отваливается",
        "тесты нестабильно падают в CI",
        "сервис не стартует после деплоя",
        "очередь задач обрабатывается с задержкой",
        "часть пользователей получает 401",
        "дублируются сообщения в обработчике",
        "таймауты при запросе к внешнему API",
    ]
    issue = rng.choice(issues)
    user = f"Помоги с разбором: {issue}. С чего начать?"
    bruno = (
        "Начни с воспроизведения и логов: сравни входные параметры, версии конфигов и тайминги. "
        "Потом сузь область проблемы до одного компонента и проверь гипотезы по очереди."
    )
    return clamp_lengths(user, bruno)


def family_business(rng: random.Random) -> Tuple[str, str]:
    scenarios = [
        "клиент недоволен задержкой поставки",
        "партнер просит скидку вне политики компании",
        "пользователь сообщает о критичном баге",
        "клиент хочет вернуть оплату",
        "команда сдвигает дедлайн",
        "нужно отказать кандидату после интервью",
        "менеджер просит срочный отчет",
        "пользователь просит приоритетную поддержку",
    ]
    scenario = rng.choice(scenarios)
    user = f"Напиши вежливый ответ: {scenario}."
    bruno = (
        "Спасибо за сообщение. Признаем проблему и понимаем неудобства. "
        "Укажи, что уже сделано, точный следующий шаг и срок обновления статуса."
    )
    return clamp_lengths(user, bruno)


def family_rewrite(rng: random.Random) -> Tuple[str, str]:
    tones = ["дружелюбном", "деловом", "уверенном", "нейтральном", "коротком"]
    texts = [
        "Мы не успеваем в срок.",
        "Сервис временно недоступен.",
        "Нам нужно больше данных для решения.",
        "Запрос принят, но есть задержка.",
        "Спасибо за обратную связь.",
    ]
    tone = rng.choice(tones)
    text = rng.choice(texts)
    user = f"Перепиши в {tone} тоне: \"{text}\""
    bruno = (
        "Предлагаю версию: \"Спасибо за терпение. Мы уже работаем над задачей и пришлем обновление в согласованный срок.\" "
        "Если хочешь, сделаю 3 альтернативы разной длины."
    )
    return clamp_lengths(user, bruno)


def family_math(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(15, 250)
    b = rng.randint(10, 90)
    user = f"Как быстро посчитать {a} * {b} в уме?"
    bruno = (
        "Разбей число на удобные части: "
        f"{a}*{b} = {a}*({b // 10}*10 + {b % 10}). "
        "Сначала посчитай десятки, потом единицы и сложи результаты."
    )
    return clamp_lengths(user, bruno)


def family_productivity(rng: random.Random) -> Tuple[str, str]:
    goals = [
        "не отвлекаться во время работы",
        "успевать больше за день",
        "сократить время на митинги",
        "системно вести заметки",
        "лучше приоритизировать задачи",
        "не выгорать на длинных проектах",
        "держать фокус на одной задаче",
    ]
    goal = rng.choice(goals)
    user = f"Дай практичный совет, как {goal}."
    bruno = (
        "Используй короткие циклы по 25-40 минут, один приоритет на цикл и обязательный итог. "
        "В конце дня фиксируй 3 результата и 1 улучшение на завтра."
    )
    return clamp_lengths(user, bruno)


def family_learning(rng: random.Random) -> Tuple[str, str]:
    topics = [
        "алгоритмы",
        "SQL",
        "системный дизайн",
        "машинное обучение",
        "Docker",
        "английский",
        "продуктовая аналитика",
    ]
    topic = rng.choice(topics)
    user = f"Как учить {topic} стабильно 30 дней?"
    bruno = (
        "Делай ежедневный цикл: теория 20 минут, практика 30 минут, краткий конспект 10 минут. "
        "Раз в неделю повторяй ошибки и обновляй план."
    )
    return clamp_lengths(user, bruno)


def build_examples(target_size: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    families: List[Callable[[random.Random], Tuple[str, str]]] = [
        family_explain,
        family_compare,
        family_plan,
        family_code,
        family_debug,
        family_business,
        family_rewrite,
        family_math,
        family_productivity,
        family_learning,
    ]

    examples: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    safety_steps = target_size * 30
    idx = 0

    while len(examples) < target_size and idx < safety_steps:
        factory = families[idx % len(families)]
        user_text, bruno_text = factory(rng)
        pair = (user_text, bruno_text)
        if pair not in seen:
            seen.add(pair)
            examples.append({"user": user_text, "bruno": bruno_text})
        idx += 1

    if len(examples) < target_size:
        raise ValueError(
            f"Could only generate {len(examples)} unique examples out of requested {target_size}."
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large instruction dataset for Bruno Prototype 0.2.")
    parser.add_argument(
        "--out",
        default="data/instruction/bruno_train_v2.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--size", type=int, default=1200, help="Number of examples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.size < 100:
        raise ValueError("--size should be at least 100 for a useful prototype dataset.")

    rows = build_examples(target_size=args.size, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
