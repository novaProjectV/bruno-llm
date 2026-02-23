# Bruno LLM

## Current Prototype

- `Bruno Prototype 0.1` (текущий опубликованный baseline)
- Следующая цель: `Bruno Prototype 0.2` (train+val, больший датасет, без truncation)

---

# Bruno Instruction Upgrade (Bruno Assistant 1)

Этот проект добавляет следующий этап после `Bruno Core 1`: инструкционное обучение (SFT), чтобы модель отвечала в формате ассистента `Пользователь / Bruno`.

## Что есть в проекте

- формат датасета `JSONL` с полями `user` и `bruno`
- `bootstrap_core1.py` для создания артефактов `Bruno Core 1`
- подготовка SFT-примеров с лоссом только на ответе Bruno
- дообучение из чекпоинта `Bruno Core 1`
- интерактивный чат для проверки

## Структура

- `bruno_core/model.py` - GPT decoder-only модель
- `bruno_core/tokenizer.py` - простой символьный токенизатор
- `scripts/bootstrap_core1.py` - создание `tokenizer.json` и `bruno_core1.pt`
- `scripts/prepare_instruction_data.py` - сборка SFT-датасета
- `scripts/train_instruction.py` - инструкционное обучение
- `scripts/chat.py` - тестовый чат
- `data/instruction/bruno_train.jsonl` - пример instruction-данных

## 1) Установите зависимости

```bash
python3 -m pip install --user -r requirements.txt
```

## 2) Подготовьте instruction-датасет

Формат одной строки `JSONL`:

```json
{"user":"Как тебя зовут?","bruno":"Я Bruno. Помогаю с кодом и задачами по тексту."}
```

Допустимые ключи: `user/bruno`, `Пользователь/Bruno`, `prompt/assistant`.

## 3) Создайте артефакты Bruno Core 1 (если их нет)

Если у вас уже есть:
- `artifacts/core1/tokenizer.json`
- `artifacts/core1/bruno_core1.pt`

этот шаг можно пропустить.

```bash
python3 scripts/bootstrap_core1.py \
  --input data/instruction/bruno_train.jsonl \
  --out-dir artifacts/core1 \
  --steps 300 \
  --block-size 256 \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 512
```

## 4) Соберите SFT-тензоры

```bash
python3 scripts/prepare_instruction_data.py \
  --input data/instruction/bruno_train.jsonl \
  --tokenizer artifacts/core1/tokenizer.json \
  --out data/processed/bruno_sft_train.pt \
  --block-size 256
```

Скрипт:
- строит шаблон `Пользователь: ... \nBruno: ...`
- считает loss только на токенах ответа Bruno
- ставит `labels=-100` для prompt-части и паддинга

## 5) Запустите инструкционное обучение

```bash
python3 scripts/train_instruction.py \
  --base-checkpoint artifacts/core1/bruno_core1.pt \
  --train-data data/processed/bruno_sft_train.pt \
  --tokenizer artifacts/core1/tokenizer.json \
  --out-dir artifacts/bruno_assistant1 \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5
```

На выходе:
- `artifacts/bruno_assistant1/checkpoint_last.pt`
- `artifacts/bruno_assistant1/checkpoint_best.pt` (если есть валидация)

## 6) Проверка в чате

```bash
python3 scripts/chat.py \
  --checkpoint artifacts/bruno_assistant1/checkpoint_last.pt \
  --tokenizer artifacts/core1/tokenizer.json
```

## Формат чекпоинта Core 1

`train_instruction.py` понимает ключи:
- конфиг: `model_config` или `model_args` или `config`
- веса: `model_state` или `model` или `state_dict`
