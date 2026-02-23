#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


DEFAULT_MAX_TOTAL_CHARS = 420
MAX_TOTAL_CHARS = DEFAULT_MAX_TOTAL_CHARS


FamilyFactory = Callable[[random.Random], Tuple[str, str]]
FamilySpec = Tuple[str, FamilyFactory, float]


def trim_with_ellipsis(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def clamp_lengths(
    user_text: str, bruno_text: str, max_total_chars: Optional[int] = None
) -> Tuple[str, str]:
    if max_total_chars is None:
        max_total_chars = MAX_TOTAL_CHARS
    user_text = user_text.strip()
    bruno_text = bruno_text.strip()
    total = len(user_text) + len(bruno_text)
    if total <= max_total_chars:
        return user_text, bruno_text

    min_user = 28
    min_bruno = 28

    # Prefer trimming assistant text first, then user text.
    bruno_limit = max(min_bruno, max_total_chars - len(user_text))
    bruno_text = trim_with_ellipsis(bruno_text, bruno_limit)

    if len(user_text) + len(bruno_text) > max_total_chars:
        user_limit = max(min_user, max_total_chars - len(bruno_text))
        user_text = trim_with_ellipsis(user_text, user_limit)

    # Hard cap if both strings are still too long.
    if len(user_text) + len(bruno_text) > max_total_chars:
        user_limit = max_total_chars // 2
        bruno_limit = max_total_chars - user_limit
        user_text = trim_with_ellipsis(user_text, user_limit)
        bruno_text = trim_with_ellipsis(bruno_text, bruno_limit)

    return user_text, bruno_text


def signed_term(value: int) -> str:
    if value >= 0:
        return f"+ {value}"
    return f"- {abs(value)}"


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


def family_math_mul_cot(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(12, 999)
    b = rng.randint(11, 99)
    tens = (b // 10) * 10
    ones = b % 10
    part1 = a * tens
    part2 = a * ones
    total = part1 + part2
    user = f"Реши: {a} * {b}"
    bruno = (
        f"Сначала умножаем {a} на {tens}: получаем {part1}. "
        f"Потом умножаем {a} на {ones}: получаем {part2}. "
        f"Складываем: {part1} + {part2} = {total}. "
        f"Ответ: {total}."
    )
    return clamp_lengths(user, bruno)


def family_math_expr_cot(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(200, 6000)
    b = rng.randint(20, 1400)
    c = rng.randint(10, 1200)
    first = a + b
    total = first - c
    user = f"Вычисли: {a} + {b} - {c}"
    bruno = (
        f"Сначала считаем сумму: {a} + {b} = {first}. "
        f"Потом вычитаем {c}: {first} - {c} = {total}. "
        f"Ответ: {total}."
    )
    return clamp_lengths(user, bruno)


def family_math_div_cot(rng: random.Random) -> Tuple[str, str]:
    divisor = rng.randint(2, 30)
    quotient = rng.randint(6, 180)
    remainder = rng.randint(0, divisor - 1)
    dividend = quotient * divisor + remainder
    base = quotient * divisor
    user = f"Найди частное и остаток: {dividend} / {divisor}"
    bruno = (
        f"Берем произведение делителя и целой части: {divisor} * {quotient} = {base}. "
        f"Вычитаем из делимого: {dividend} - {base} = {remainder}. "
        f"Ответ: частное {quotient}, остаток {remainder}."
    )
    return clamp_lengths(user, bruno)


def family_math_percent_cot(rng: random.Random) -> Tuple[str, str]:
    base = rng.randint(2, 300) * 100
    pct = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75])
    one_percent = base // 100
    total = one_percent * pct
    user = f"Найди {pct}% от {base}."
    bruno = (
        f"Находим 1%: {base} / 100 = {one_percent}. "
        f"Умножаем на {pct}: {one_percent} * {pct} = {total}. "
        f"Ответ: {total}."
    )
    return clamp_lengths(user, bruno)


def family_math_equation_cot(rng: random.Random) -> Tuple[str, str]:
    a = rng.randint(2, 16)
    x = rng.randint(-30, 30)
    b = rng.randint(-60, 60)
    c = a * x + b
    user = f"Реши уравнение: {a}x {signed_term(b)} = {c}"
    rhs = c - b
    if b >= 0:
        step = f"Переносим {b} вправо: {a}x = {c} - {b} = {rhs}."
    else:
        step = f"Переносим {-b} вправо со знаком плюс: {a}x = {c} + {abs(b)} = {rhs}."
    bruno = (
        f"{step} "
        f"Делим обе части на {a}: x = {rhs} / {a} = {x}. "
        f"Ответ: x = {x}."
    )
    return clamp_lengths(user, bruno)


def family_math_word_cot(rng: random.Random) -> Tuple[str, str]:
    box_count = rng.randint(3, 35)
    per_box = rng.randint(6, 48)
    extra = rng.randint(5, 120)
    total = box_count * per_box + extra
    user = (
        f"В магазине {box_count} коробок по {per_box} тетрадей и еще {extra} отдельных тетрадей. "
        "Сколько тетрадей всего?"
    )
    bruno = (
        f"Сначала считаем тетради в коробках: {box_count} * {per_box} = {box_count * per_box}. "
        f"Добавляем отдельные тетради: {box_count * per_box} + {extra} = {total}. "
        f"Ответ: {total}."
    )
    return clamp_lengths(user, bruno)


def family_math_quick_answer(rng: random.Random) -> Tuple[str, str]:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(10, 999)
    b = rng.randint(2, 120)
    if op == "+":
        result = a + b
    elif op == "-":
        if a < b:
            a, b = b, a
        result = a - b
    else:
        result = a * b
    user = f"Посчитай быстро: {a} {op} {b}"
    bruno = f"Ответ: {result}."
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


def family_security(rng: random.Random) -> Tuple[str, str]:
    threats = [
        "SQL-инъекции",
        "XSS",
        "CSRF",
        "утечка токенов доступа",
        "подбор паролей",
        "небезопасная загрузка файлов",
        "ошибка авторизации по ролям",
        "exposed secrets в репозитории",
    ]
    threat = rng.choice(threats)
    user = f"Как защитить сервис от {threat}?"
    bruno = (
        "Сделай защиту в слоях: валидация входа, строгая авторизация, лимиты, логирование и алерты. "
        "Проверь это тестом и коротким security-чеклистом перед релизом."
    )
    return clamp_lengths(user, bruno)


def family_system_design(rng: random.Random) -> Tuple[str, str]:
    systems = [
        "чат в реальном времени",
        "уведомления по событиям",
        "поиск по каталогу товаров",
        "очередь фоновых задач",
        "сервис рекомендаций",
        "файловое хранилище",
        "аналитику пользовательских событий",
    ]
    system = rng.choice(systems)
    user = f"С чего начать проектировать {system}?"
    bruno = (
        "Начни с требований и ограничений, затем опиши компоненты, поток данных и SLA. "
        "После этого выбери хранилище, стратегию масштабирования и план наблюдаемости."
    )
    return clamp_lengths(user, bruno)


def family_language_explain_audience(rng: random.Random) -> Tuple[str, str]:
    topics = [
        "A/B тесты",
        "асинхронные очереди",
        "наблюдаемость сервиса",
        "кредитный скоринг",
        "прогнозирование спроса",
        "архитектуру событий",
        "управление инцидентами",
        "приоритизацию бэклога",
        "пользовательский retention",
        "границы микросервисов",
        "версионирование API",
        "регрессионное тестирование",
        "категоризацию тикетов",
        "аналитику продукта",
        "управление рисками",
        "запуск MVP",
        "юнит-экономику",
        "обратную связь от клиентов",
        "UX-исследования",
        "работу с гипотезами",
        "построение roadmap",
        "гигиену данных",
        "автоматизацию отчетов",
        "согласование требований",
        "процессы релиза",
        "модель монетизации",
        "управление SLA",
        "модель ролей и доступов",
        "канбан-подход",
        "декомпозицию задач",
        "экспериментальный дизайн",
        "pull-request процесс",
        "code review культуру",
        "продуктовые метрики",
        "time-to-market",
        "управление изменениями",
        "интеграции с партнерами",
        "работу с логами",
        "data governance",
        "фиче-флаги",
        "rate limiting",
        "поиск узких мест",
        "capacity planning",
        "обработку ошибок",
        "снижение техдолга",
        "качество документации",
    ]
    audiences = [
        "новичка в команде",
        "менеджера без технического бэкграунда",
        "студента 1 курса",
        "junior-разработчика",
        "нового аналитика",
        "владельца малого бизнеса",
        "клиента на демо",
        "стажера в отделе",
    ]
    formats = [
        "в 3 коротких пунктах",
        "простым языком без жаргона",
        "через понятную аналогию",
        "с одним практичным примером",
        "с акцентом на типичные ошибки",
        "как мини-памятку",
    ]
    topic = rng.choice(topics)
    audience = rng.choice(audiences)
    fmt = rng.choice(formats)
    examples_count = rng.randint(1, 4)
    user = f"Объясни {topic} для {audience} {fmt}, добавь {examples_count} примера."
    bruno = (
        f"{topic} — это способ улучшить предсказуемость результата. "
        "Сначала определяем цель и метрику успеха, затем шаги внедрения, "
        "после этого проверяем результат на реальном сценарии."
    )
    return clamp_lengths(user, bruno)


def family_language_draft_message(rng: random.Random) -> Tuple[str, str]:
    situations = [
        "клиент просит ускорить релиз",
        "партнер сообщил о критичном баге",
        "команда переносит срок на неделю",
        "пользователь недоволен качеством поддержки",
        "клиенту нужно уточнение требований",
        "мы отклоняем запрос на нестандартную скидку",
        "проект временно заморожен",
        "нужно согласовать новый план работ",
        "в релизе найден регресс",
        "часть функций будет недоступна ночью",
        "мы завершили расследование инцидента",
        "нужно извиниться за задержку",
        "клиент спрашивает про статус задачи",
        "пользователь просит возврат средств",
        "мы закрываем старый тариф",
        "вводим новое ограничение API",
        "обновляем политику безопасности",
        "нужно предложить альтернативное решение",
        "просим дополнительную информацию по кейсу",
        "планируем миграцию данных",
    ]
    tones = [
        "дружелюбном",
        "деловом",
        "нейтральном",
        "уверенном",
        "эмпатичном",
        "максимально кратком",
    ]
    channels = ["email", "чат", "служебное сообщение", "ответ в тикете", "сообщение в CRM"]
    constraints = [
        "с четким следующим шагом",
        "с конкретным сроком ответа",
        "с акцентом на ответственность команды",
        "без лишних деталей",
        "с предложением созвона",
        "с пунктами 1-2-3",
        "с подтверждением, что запрос получен",
        "с понятным планом действий",
    ]
    situation = rng.choice(situations)
    tone = rng.choice(tones)
    channel = rng.choice(channels)
    constraint = rng.choice(constraints)
    update_hours = rng.randint(2, 72)
    user = (
        f"Напиши сообщение в {channel} в {tone} тоне: {situation}, {constraint}. "
        f"Обновление статуса через {update_hours} часов."
    )
    bruno = (
        "Спасибо за сообщение. Мы приняли задачу в работу и уже проверяем детали. "
        "Следующий апдейт отправим в обозначенный срок, а при изменениях предупредим заранее."
    )
    return clamp_lengths(user, bruno)


def family_language_goal_plan(rng: random.Random) -> Tuple[str, str]:
    goals = [
        "поднять качество кода в проекте",
        "снизить количество багов после релиза",
        "ускорить onboarding новых сотрудников",
        "улучшить стабильность сервиса",
        "сократить время решения инцидентов",
        "повысить конверсию регистрации",
        "уменьшить churn в продукте",
        "повысить скорость фиче-разработки",
        "улучшить качество технической документации",
        "уменьшить нагрузку на поддержку",
        "внедрить регулярные ретроспективы",
        "повысить дисциплину планирования",
        "сделать процессы прозрачнее",
        "вывести метрики в единый дашборд",
        "сократить время ответа API",
        "стандартизировать code review",
        "системно работать с обратной связью",
        "подготовить продукт к масштабированию",
        "оптимизировать расходы на инфраструктуру",
        "улучшить прогнозирование сроков",
    ]
    horizons = ["за 2 недели", "за месяц", "за 6 недель", "за квартал"]
    limits = [
        "с минимальными затратами",
        "без увеличения команды",
        "без остановки текущих релизов",
        "с одним ответственным на поток",
        "с приоритетом на быстрый результат",
        "в рамках текущего бэклога",
    ]
    goal = rng.choice(goals)
    horizon = rng.choice(horizons)
    limit = rng.choice(limits)
    target_improvement = rng.randint(10, 60)
    user = (
        f"Составь практичный план, как {goal} {horizon} {limit}. "
        f"Цель: улучшить ключевую метрику на {target_improvement}%."
    )
    bruno = (
        "1) Зафиксируй базовую метрику и целевое значение. "
        "2) Определи 3 приоритетных действия с владельцами и сроками. "
        "3) Раз в неделю оценивай прогресс и корректируй план по фактам."
    )
    return clamp_lengths(user, bruno)


def get_family_specs(profile: str) -> List[FamilySpec]:
    generic: List[FamilySpec] = [
        ("explain", family_explain, 1.0),
        ("compare", family_compare, 1.0),
        ("plan", family_plan, 1.0),
        ("code", family_code, 1.0),
        ("debug", family_debug, 1.0),
        ("business", family_business, 1.0),
        ("rewrite", family_rewrite, 1.0),
        ("math_tips", family_math, 1.0),
        ("productivity", family_productivity, 1.0),
        ("learning", family_learning, 1.0),
        ("security", family_security, 1.0),
        ("system_design", family_system_design, 1.0),
    ]

    if profile == "generic":
        return generic
    if profile == "v4-cot":
        return [
            ("math_mul_cot", family_math_mul_cot, 1.8),
            ("math_expr_cot", family_math_expr_cot, 1.7),
            ("math_div_cot", family_math_div_cot, 1.5),
            ("math_percent_cot", family_math_percent_cot, 1.2),
            ("math_equation_cot", family_math_equation_cot, 1.5),
            ("math_word_cot", family_math_word_cot, 1.8),
            ("math_quick_answer", family_math_quick_answer, 0.9),
            ("math_tips", family_math, 0.5),
            ("lang_explain_audience", family_language_explain_audience, 3.8),
            ("lang_draft_message", family_language_draft_message, 3.6),
            ("lang_goal_plan", family_language_goal_plan, 3.6),
            ("explain", family_explain, 0.8),
            ("compare", family_compare, 0.8),
            ("plan", family_plan, 0.7),
            ("code", family_code, 0.8),
            ("debug", family_debug, 0.8),
            ("business", family_business, 0.6),
            ("rewrite", family_rewrite, 0.6),
            ("productivity", family_productivity, 0.6),
            ("learning", family_learning, 0.6),
            ("security", family_security, 0.7),
            ("system_design", family_system_design, 0.7),
        ]
    raise ValueError(f"Unknown profile: {profile}")


def build_examples(target_size: int, seed: int, profile: str) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    rng = random.Random(seed)
    family_specs = get_family_specs(profile)
    names = [name for name, _, _ in family_specs]
    weights = [weight for _, _, weight in family_specs]
    factories = [factory for _, factory, _ in family_specs]

    examples: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    family_counts: Dict[str, int] = defaultdict(int)
    safety_steps = target_size * 40
    attempts = 0

    while len(examples) < target_size and attempts < safety_steps:
        choice_idx = rng.choices(range(len(factories)), weights=weights, k=1)[0]
        factory = factories[choice_idx]
        user_text, bruno_text = factory(rng)
        pair = (user_text, bruno_text)
        if pair not in seen:
            seen.add(pair)
            examples.append({"user": user_text, "bruno": bruno_text})
            family_counts[names[choice_idx]] += 1
        attempts += 1

    if len(examples) < target_size:
        raise ValueError(
            f"Could only generate {len(examples)} unique examples out of requested {target_size}."
        )
    return examples, dict(family_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large instruction dataset for Bruno prototypes.")
    parser.add_argument(
        "--out",
        default="data/instruction/bruno_train_v2.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--size", type=int, default=1200, help="Number of examples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--profile",
        choices=["generic", "v4-cot"],
        default="generic",
        help="Dataset family profile.",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=DEFAULT_MAX_TOTAL_CHARS,
        help="Max characters for user+assistant text before clipping.",
    )
    args = parser.parse_args()

    if args.size < 100:
        raise ValueError("--size should be at least 100 for a useful prototype dataset.")
    if args.max_total_chars < 120:
        raise ValueError("--max-total-chars should be >= 120.")

    global MAX_TOTAL_CHARS
    MAX_TOTAL_CHARS = args.max_total_chars

    rows, family_counts = build_examples(target_size=args.size, seed=args.seed, profile=args.profile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} examples to {out_path}")
    print("Profile:", args.profile)
    print("Family summary:", json.dumps(dict(sorted(family_counts.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
