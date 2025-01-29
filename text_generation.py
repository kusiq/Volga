from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, model, tokenizer, max_length=100):
    # Токенизация входного текста
    inputs = tokenizer(prompt, return_tensors='pt')

    # Генерация текста
    outputs = model.generate(
        inputs['input_ids'].to(model.device),
        max_length=max_length,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    # Декодирование сгенерированного текста
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text