if __name__ == "__main__":
    from data_preparation import load_and_preprocess_data
    from model_training import train_model
    from text_generation import generate_text
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    # Загрузка и предобработка данных
    full_text = load_and_preprocess_data()

    # Проверка, существует ли уже обученная модель
    if not os.path.exists("./universal_model"):
        print("Обучение модели...")
        model, tokenizer = train_model(full_text)
    else:
        print("Загрузка сохраненной модели и токенизатора...")
        model = AutoModelForCausalLM.from_pretrained("./universal_model", device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("./universal_model")

    # Интерактивный режим для генерации текста
    print("\nМодель 'Vолга' готова к генерации текста!")
    while True:
        prompt = input("\nВведите начало текста ('exit' или 'выход' для завершения работы): ")
        if prompt.lower() in ['exit', 'выход']:
            print("Завершение работы.")
            break

        # Генерация текста
        generated_text = generate_text(prompt, model, tokenizer, max_length=100)
        print(f"\nСгенерированный текст:\n{generated_text}\n")