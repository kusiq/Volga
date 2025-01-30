import wikipediaapi
import re

def load_and_preprocess_data():
    wiki = wikipediaapi.Wikipedia(
        language='ru',
        user_agent='MyProject/1.0 (matveykuskov@gmail.com)'
    )
    # Загрузка большего количества статей
    page_titles = [
        "Александр Сергеевич Пушкин", "Русский язык", "Космос", "Химия",
        "Китай", "Украина", "США", "История", "Москва", "Кострома",
        "Санкт-Петербург", "История России", "Литература", "Культура",
        "Наука", "Технологии", "Искусство", "Философия", "Математика",
        "Биология", "География", "Экономика", "Политика", "Спорт"
    ]

    texts = []

    for title in page_titles:
        page = wiki.page(title)
        if page.exists():
            text = re.sub(r'\[\d+\]', '', page.text)  # Удаление сносок
            texts.append(text)

    full_text = "\n".join(texts)[:1_000_000]

    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:]', '', text)  # Удаление специальных символов
        text = text.strip()

        return text

    full_text = preprocess_text(full_text)
    
    print(f"Total length of text: {len(full_text)} characters")

    return full_text