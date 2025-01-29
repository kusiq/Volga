from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model, tokenizer, max_length=50):
    # Токенизация входного текста
    inputs = tokenizer(prompt, return_tensors='pt')

    # Генерация текста
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id  # Установка pad_token_id
    )

    # Декодирование сгенерированного текста
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text