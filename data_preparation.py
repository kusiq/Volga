from datasets import load_dataset
import re

def load_and_preprocess_data():
    # Загрузка датасета
    ds = load_dataset("abobster/pushkin_new")

    # Функция для предобработки текста
    def preprocess_text(text):
        # Удаление лишних пробелов и символов
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    # Применение предобработки ко всему тексту
    train_texts = [preprocess_text(example['text']) for example in ds['train']]
    test_texts = [preprocess_text(example['text']) for example in ds['test']] if 'test' in ds else []

    # Объединение всех текстов в один большой текст
    full_text = '\n'.join(train_texts + test_texts)

    print(f"Total length of text: {len(full_text)} characters")

    return full_text