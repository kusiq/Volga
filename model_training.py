from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from data_preparation import load_and_preprocess_data

# Класс для создания пользовательского датасета
class TextDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=512):
        self.tokenizer = tokenizer
        self.text = text
        self.max_length = max_length

    def __len__(self):
        return len(self.text) // (self.max_length - 1)

    def __getitem__(self, idx):
        start = idx * (self.max_length - 1)
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
    model_name = "gpt2-medium"  # Используйте более крупную версию модели

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
        num_train_epochs=3,  # Уменьшите количество эпох для отладки
        per_device_train_batch_size=4,  # Уменьшите размер батча
        save_steps=10_000,
        save_total_limit=2,
        use_cpu=True  # Использовать только CPU !!!!!!!!!
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

if __name__ == "__main__":
    full_text = load_and_preprocess_data()
    train_model(full_text)