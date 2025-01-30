if __name__ == "__main__":
    from data_preparation import load_and_preprocess_data
    from model_training import train_model
    from text_generation import generate_text
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    # Загрузка и предобработка данных
    full_text = load_and_preprocess_data()

    # Проверка, существует ли уже обученная модель
    if not os.path.exists("./volga_v0.3"):
        print("Обучение модели...")
        model, tokenizer = train_model(full_text)
    else:
        print("Загрузка сохраненной модели и токенизатора...")
        model = AutoModelForCausalLM.from_pretrained("./volga_v0.3")
        tokenizer = AutoTokenizer.from_pretrained("./volga_v0.3")

    # Интерактивный режим для генерации текста
    print("\nМодель готова к генерации текста!")

    while True:
        prompt = input("\nВведите начало текста (или 'exit' для выхода): ")

        if prompt.lower() in ['exit', 'quit']:
            print("Завершение работы.")
            break

        # Генерация текста
        generated_text = generate_text(prompt, model, tokenizer, max_length=200, temperature=0.7, top_k=50)
        print(f"\nСгенерированный текст:\n{generated_text}\n")