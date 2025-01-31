import importlib

def check_dependencies():
    required_libraries = ["datasets", "transformers", "torch", "bitsandbytes"]
    missing_libraries = []
    for library in required_libraries:
        try:
            importlib.import_module(library)
        except ImportError:
            missing_libraries.append(library)
    if missing_libraries:
        raise ImportError(f"Необходимо установить следующие библиотеки: {', '.join(missing_libraries)}")

if __name__ == "__main__":
    # Проверка зависимостей
    check_dependencies()
    
    from data_preparation import load_and_preprocess_data
    from model_training import train_model
    from text_generation import generate_text
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    import torch

    # Загрузка и предобработка данных
    train_text, val_text = load_and_preprocess_data()

    # Проверка, существует ли уже обученная модель
    if not os.path.exists("./volga_v1.3"):
        print("Обучение модели...")
        model, tokenizer = train_model(train_text, val_text)
    else:
        print("Загрузка сохраненной модели и токенизатора...")
        model = AutoModelForCausalLM.from_pretrained("./volga_v1.3")
        tokenizer = AutoTokenizer.from_pretrained("./volga_v1.3")

    # Перемещение модели на устройство (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Интерактивный режим для генерации текста
    print("\nМодель готова к генерации текста!")
    while True:
        prompt = input("\nВведите начало текста (или 'exit' для выхода): ")
        if prompt.lower() in ['exit', 'quit']:
            print("Завершение работы.")
            break
        
        # Генерация текста
        generated_text = generate_text(prompt, model, tokenizer, max_length=500, temperature=0.7, top_k=50)

        print(f"\nСгенерированный текст:\n{generated_text}\n")