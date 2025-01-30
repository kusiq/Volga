from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, model, tokenizer, max_length=200, temperature=0.7, top_k=50):
    # Перемещение модели на устройство (GPU или CPU)
    device = next(model.parameters()).device  # Определяем устройство модели
    inputs = tokenizer(prompt, return_tensors='pt')  # Токенизация входного текста

    # Перемещение входных данных на устройство модели
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Генерация текста
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Включаем режим выборки
        temperature=temperature,  # Уменьшение температуры делает ответы более детерминированными
        top_k=top_k,  # Ограничивает выборку до top_k наиболее вероятных токенов
    )

    # Декодирование сгенерированного текста
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text