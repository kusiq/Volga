from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

def validate_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке модели или токенизатора: {e}")

def validate_training_args(training_args):
    if training_args.per_device_train_batch_size > 32:
        raise ValueError("Размер батча слишком большой. Уменьшите его до разумного значения.")
    if training_args.num_train_epochs <= 0:
        raise ValueError("Количество эпох должно быть больше нуля.")
    if training_args.fp16 and not torch.cuda.is_available():
        raise ValueError("Mixed Precision (fp16) требует GPU. Переключитесь на CPU или исправьте конфигурацию.")

class TextDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Токенизация всего текста
        tokenized_text = tokenizer(
            text,
            return_tensors="pt",
            max_length=None,  # Не ограничиваем длину заранее
            truncation=False,
            padding=False
        )["input_ids"][0]

        # Разделение токенизированного текста на чанки
        self.chunks = [tokenized_text[i:i + max_length] for i in range(0, len(tokenized_text), max_length)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        # Дополнение последнего чанка до max_length
        if len(chunk) < self.max_length:
            padding_length = self.max_length - len(chunk)
            chunk = torch.cat([chunk, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])

        attention_mask = (chunk != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": chunk,
            "attention_mask": attention_mask,
            "labels": chunk.clone()  # Для языковых моделей labels совпадают с input_ids
        }

def train_model(train_text, val_text):
    model_name = "gpt2"

    # Проверка модели и токенизатора
    tokenizer, model = validate_model_and_tokenizer(model_name)

    # Перемещение модели на GPU, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = TextDataset(tokenizer, train_text)
    val_dataset = TextDataset(tokenizer, val_text)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        save_steps=10_000,
        gradient_accumulation_steps=2,  # Эффективный батч: 8 × 2 = 16
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
        logging_dir="./logs",
        logging_steps=500,
        eval_strategy="steps",  # Добавляем валидацию каждые N шагов
        eval_steps=500,               # Частота валидации
    )

    # Проверка параметров обучения
    validate_training_args(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Добавляем валидационный набор
    )
    trainer.train()
    model.save_pretrained("./volga_v0.3")
    tokenizer.save_pretrained("./volga_v0.3")
    return model, tokenizer