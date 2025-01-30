from datasets import load_dataset
import re
import random

def load_and_preprocess_data():
    # Загрузка датасета wikimedia/wikipedia на русском языке (последняя версия)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", trust_remote_code=True)

    # Ограничение до 0.1% статей для отладки
    total_articles = len(dataset)
    sample_size = int(total_articles * 0.001)  # 0.1% от общего количества статей
    sampled_indices = random.sample(range(total_articles), sample_size)
    sampled_dataset = [dataset[i] for i in sampled_indices]

    # Сбор текстов статей
    texts = []
    for article in sampled_dataset:
        texts.append(article['text'])

    # Объединение текстов в один большой текст
    full_text = "\n".join(texts)
    # Ограничение текста до ~500 тысяч символов для увеличения объема данных
    full_text = full_text[:500_000]

    # Предобработка текста
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
        text = text.strip()
        return text

    full_text = preprocess_text(full_text)
    print(f"Total length of text: {len(full_text)} characters")
    return full_text