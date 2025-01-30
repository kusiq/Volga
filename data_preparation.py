from datasets import load_dataset
import re

def load_and_preprocess_data():
    # Загрузка новых датасетов
    print("Загрузка датасетов...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train[:1%]")  # Russian Wikipedia Dump

    # Функция для предобработки текста
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
        text = text.strip()              # Удаление пробелов в начале и конце
        return text

    # Объединение текстов из всех датасетов
    train_texts = (
        [preprocess_text(example['text']) for example in ds]
    )

    # Объединение всех текстов в один большой текст
    full_text = '\n'.join(train_texts)

    print(f"Total length of text: {len(full_text)} characters")
    
    return full_text