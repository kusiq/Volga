if __name__ == "__main__":
    from data_preparation import load_and_preprocess_data
    from model_training import train_model
    from text_generation import generate_text
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import os

    # Загрузка и предобработка данных
    full_text = load_and_preprocess_data()

    # Проверка, существует ли уже обученная модель
    if not os.path.exists("./pushkin_gpt2"):
        print("Обучение модели...")
        model, tokenizer = train_model(full_text)
    else:
        print("Загрузка сохраненной модели и токенизатора...")
        model = GPT2LMHeadModel.from_pretrained("./pushkin_gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("./pushkin_gpt2")

    # Интерактивный режим для генерации текста
    print("\nМодель готова к генерации текста!")
    while True:
        prompt = input("\nВведите начало текста (или 'exit' для выхода): ")
        if prompt.lower() in ['exit', 'quit']:
            print("Завершение работы.")
            break

        # Генерация текста
        generated_text = generate_text(prompt, model, tokenizer, max_length=50)
        print(f"\nСгенерированный текст:\n{generated_text}\n")