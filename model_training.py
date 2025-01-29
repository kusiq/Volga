from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
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
            "labels": inputs["input_ids"].squeeze(0)
        }

def train_model(full_text):
    # Загрузка предобученного токенизатора и модели Qwen-7B
    model_name = "Qwen/Qwen-7B"  # Вы можете заменить на другую модель, например "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Использование 8-битного обучения с bitsandbytes
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Автоматическое распределение по устройствам
        load_in_8bit=True,  # Включение 8-битного обучения
        torch_dtype=torch.float16  # Использование 16-битной точности
    )

    # Установка pad_token
    tokenizer.pad_token = tokenizer.eos_token

    # Настройка LoRA (Low-Rank Adaptation)
    lora_config = LoraConfig(
        r=8,  # Ранг матрицы (меньше = быстрее)
        lora_alpha=32,  # Коэффициент масштабирования
        target_modules=["q_proj", "v_proj"],  # Целевые слои для адаптации
        lora_dropout=0.1,  # Dropout для LoRA
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Создание пользовательского датасета
    dataset = TextDataset(tokenizer, full_text)

    # Определение аргументов для обучения
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16=True,  # Использование 16-битной точности
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
    model.save_pretrained("./universal_model")
    tokenizer.save_pretrained("./universal_model")

    return model, tokenizer