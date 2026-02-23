# Bruno LLM

## Prototype Versions

- `Bruno Prototype 0.1`  
  baseline checkpoint: `artifacts/bruno_assistant1/checkpoint_last.pt`
- `Bruno Prototype 0.2`  
  цель: `train_loss + val_loss`, увеличенный датасет, без truncation

## Files

- `bruno_core/model.py` - GPT decoder-only модель
- `bruno_core/tokenizer.py` - простой символьный токенизатор
- `scripts/bootstrap_core1.py` - создание базового `Core 1` чекпоинта
- `scripts/generate_prototype_dataset.py` - генерация большого instruction-датасета
- `scripts/prepare_instruction_data.py` - подготовка `.pt` тензоров train/val
- `scripts/train_instruction.py` - инструкционное обучение (SFT)
- `scripts/chat.py` - интерактивный чат

## Setup

```bash
python3 -m pip install --user -r requirements.txt
```

## Publish Bruno Prototype 0.1

```bash
git add README.md .gitignore requirements.txt bruno_core scripts data/instruction/bruno_train.jsonl \
  artifacts/core1/tokenizer.json artifacts/bruno_assistant1/checkpoint_last.pt artifacts/bruno_assistant1/train_history.json
git commit -m "release: Bruno Prototype 0.1"
git tag -a bruno-prototype-0.1 -m "Bruno Prototype 0.1"
git push -u origin main
git push origin bruno-prototype-0.1
```

## Build Bruno Prototype 0.2

### 1) Сгенерируйте большой датасет

```bash
python3 scripts/generate_prototype_dataset.py \
  --out data/instruction/bruno_train_v2.jsonl \
  --size 1200 \
  --seed 42
```

### 2) Создайте новый Core 1 для 0.2

```bash
python3 scripts/bootstrap_core1.py \
  --input data/instruction/bruno_train_v2.jsonl \
  --out-dir artifacts/prototype_0_2/core1 \
  --steps 400 \
  --block-size 512 \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 512
```

### 3) Подготовьте train/val без truncation

```bash
python3 scripts/prepare_instruction_data.py \
  --input data/instruction/bruno_train_v2.jsonl \
  --tokenizer artifacts/prototype_0_2/core1/tokenizer.json \
  --out data/processed/prototype_0_2_train.pt \
  --out-val data/processed/prototype_0_2_val.pt \
  --val-ratio 0.1 \
  --block-size 512 \
  --truncate-mode error \
  --auto-block-size
```

`--truncate-mode error` гарантирует, что длинные примеры не будут тихо обрезаны.

### 4) Обучите SFT с train+val

```bash
python3 scripts/train_instruction.py \
  --base-checkpoint artifacts/prototype_0_2/core1/bruno_core1.pt \
  --train-data data/processed/prototype_0_2_train.pt \
  --val-data data/processed/prototype_0_2_val.pt \
  --tokenizer artifacts/prototype_0_2/core1/tokenizer.json \
  --out-dir artifacts/prototype_0_2/assistant \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --prototype-name "Bruno Prototype 0.2"
```

На каждой эпохе будет одновременно вывод:
- `train_loss`, `train_ppl`
- `val_loss`, `val_ppl`

### 5) Проверка в чате

```bash
python3 scripts/chat.py \
  --checkpoint artifacts/prototype_0_2/assistant/checkpoint_last.pt \
  --tokenizer artifacts/prototype_0_2/core1/tokenizer.json
```
