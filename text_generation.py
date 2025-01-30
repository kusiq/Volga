from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, model, tokenizer, max_length=1000, temperature=0.7, top_k=50):
    inputs = tokenizer(prompt, return_tensors='pt')

    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,  # Уменьшение температуры делает ответы более детерминированными
        top_k=top_k,  # Ограничивает выборку до top_k наиболее вероятных токенов
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text