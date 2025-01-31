from datasets import load_dataset
import re
import random

def validate_data(text):
    if not text or len(text) < 1000:
        raise ValueError("Текст слишком короткий или пустой.")
    if not isinstance(text, str):
        raise TypeError("Данные должны быть строкой.")

def load_and_preprocess_data():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", trust_remote_code=True)
    train_test_split = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    validation_dataset = train_test_split['test']
    
    total_train_articles = len(train_dataset)
    total_val_articles = len(validation_dataset)
    
    sample_size_train = int(total_train_articles * 0.2)  # 20%
    sample_size_val = int(total_val_articles * 0.2)      # 20%
    
    sampled_train_indices = random.sample(range(total_train_articles), sample_size_train)
    sampled_val_indices = random.sample(range(total_val_articles), sample_size_val)
    
    sampled_train_dataset = [train_dataset[i] for i in sampled_train_indices]
    sampled_val_dataset = [validation_dataset[i] for i in sampled_val_indices]
    
    def collect_texts(sampled_dataset):
        texts = []
        for article in sampled_dataset:
            texts.append(article['text'])
        return "\n".join(texts)
    
    train_text = collect_texts(sampled_train_dataset)
    val_text = collect_texts(sampled_val_dataset)
    
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    train_text = preprocess_text(train_text[:100_000_000])  # 100 млн символов
    val_text = preprocess_text(val_text[:10_000_000])       # 10 млн символов
    
    print(f"Total length of train text: {len(train_text)} characters")
    print(f"Total length of validation text: {len(val_text)} characters")
    
    validate_data(train_text)
    validate_data(val_text)
    
    return train_text, val_text