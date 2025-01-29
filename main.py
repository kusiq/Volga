if __name__ == "__main__":
    from data_preparation import load_and_preprocess_data
    from model_training import train_model
    from text_generation import generate_text

    # Загрузка и предобработка данных
    full_text = load_and_preprocess_data()

    # Обучение модели
    model, tokenizer = train_model(full_text)

    # Генерация текста
    prompt = "Так и мне узнать случилось,"
    generated_text = generate_text(prompt, model, tokenizer)
    print(generated_text)