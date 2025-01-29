from datasets import load_dataset
import re

def load_and_preprocess_data():
    # Загрузка новых датасетов
    print("Загрузка датасетов...")
    ds1 = load_dataset("wikipedia", language="ru", date="20250101", split="train[:1%]", trust_remote_code=True)  # Russian Wikipedia Dump

    # Функция для предобработки текста
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
        text = text.strip()              # Удаление пробелов в начале и конце
        return text

    # Объединение текстов из всех датасетов
    train_texts = (
        [preprocess_text(example['text']) for example in ds1]
    )

    # Объединение всех текстов в один большой текст
    full_text = '\n'.join(train_texts)

    print(f"Total length of text: {len(full_text)} characters")
    return full_text