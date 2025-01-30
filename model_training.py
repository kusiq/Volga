from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Класс для создания пользовательского датасета
class TextDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=1024):
        self.tokenizer = tokenizer
        self.text = text
        self.max_length = max_length

    def __len__(self):
        return len(self.text) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        text_chunk = self.text[start:end]
        inputs = self.tokenizer(
            text_chunk,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0)  # Для языковых моделей labels совпадают с input_ids
        }

def train_model(full_text):
    # Загрузка предобученного токенизатора и модели
    model_name = "openai-community/gpt2"  # Пример модели с ~500 млн параметров <button class="citation-flag" data-index="2">
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Установка pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # Создание пользовательского датасета
    dataset = TextDataset(tokenizer, full_text)

    # Определение аргументов для обучения
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Создание тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Обучение модели
    trainer.train()

    # Сохранение модели и токенизатора
    print("Сохранение модели и токенизатора...")
    model.save_pretrained("./volga_v0.3")
    tokenizer.save_pretrained("./volga_v0.3")

    return model, tokenizer