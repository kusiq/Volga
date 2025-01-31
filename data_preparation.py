from datasets import load_dataset
import re
import random

def validate_data(text):
    if not text or len(text) < 1000:
        raise ValueError("Текст слишком короткий или пустой.")
    if not isinstance(text, str):
        raise TypeError("Данные должны быть строкой.")

def load_and_preprocess_data():
    # Загрузка Simple Wikipedia
    dataset = load_dataset("pszemraj/simple_wikipedia")
    
    # Разделение данных на train и validation
    train_test_split = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    validation_dataset = train_test_split['test']
    
    # Сбор текстов статей
    def collect_texts(sampled_dataset):
        texts = []
        for article in sampled_dataset:
            texts.append(preprocess_text(article['text']))
        return "\n".join(texts)
    
    train_text = collect_texts(train_dataset)
    val_text = collect_texts(validation_dataset)
    
    # Предобработка текста
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    # Ограничение объёма данных
    train_text = preprocess_text(train_text[:50_000_000])  # 10 млн символов
    val_text = preprocess_text(val_text[:5_000_000])      # 1 млн символов
    
    print(f"Total length of train text: {len(train_text)} characters")
    print(f"Total length of validation text: {len(val_text)} characters")
    
    validate_data(train_text)
    validate_data(val_text)
    
    return train_text, val_text