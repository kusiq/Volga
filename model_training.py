from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from bitsandbytes.optim import AdamW8bit  # Используем 8-bit Adam
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

class TextDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=512):  # Уменьшаем max_length до 512
        self.tokenizer = tokenizer
        self.max_length = max_length
        tokenized_text = tokenizer(
            text,
            return_tensors="pt",
            max_length=None,
            truncation=False,
            padding=False
        )["input_ids"][0]
        self.chunks = [tokenized_text[i:i + max_length] for i in range(0, len(tokenized_text), max_length)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        if len(chunk) < self.max_length:
            padding_length = self.max_length - len(chunk)
            chunk = torch.cat([chunk, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
        attention_mask = (chunk != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": chunk,
            "attention_mask": attention_mask,
            "labels": chunk.clone()
        }

def train_model(train_text, val_text):
    model_name = "openai-community/gpt2-medium"
    tokenizer, model = validate_model_and_tokenizer(model_name)
    
    # Перемещение модели на GPU, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.cuda.empty_cache()
    
    train_dataset = TextDataset(tokenizer, train_text, max_length=512)
    val_dataset = TextDataset(tokenizer, val_text, max_length=512)
    
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=2,  # Уменьшаем количество эпох
        per_device_train_batch_size=4,  # Уменьшаем размер батча
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Компенсируем меньший батч
        save_steps=500,
        logging_steps=100,
        eval_steps=200,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,  # Используем Mixed Precision
        optim="adamw_bnb_8bit",  # Используем 8-bit Adam
        dataloader_num_workers=2,
        tf32=True,
        max_grad_norm=1.0
    )
    
    optimizer = AdamW8bit(model.parameters(), lr=2e-4)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, None),
    )
    
    trainer.train()

    model.save_pretrained("./volga_v1.3")
    tokenizer.save_pretrained("./volga_v1.3")
    
    return model, tokenizer