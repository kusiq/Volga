from huggingface_hub import HfApi

# Создаем объект API для работы с Hugging Face Hub
api = HfApi()

# Загружаем папку с моделью на Hugging Face Hub
api.upload_folder(
    folder_path="./pushkin_gpt2",  # Путь к папке с сохраненной моделью
    repo_id="kusiq/volga",  # Например, "username/pushkin-gpt2"
    repo_type="model"
)