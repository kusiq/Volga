from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(prompt, model, tokenizer, max_length=500, temperature=0.7, top_k=50):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=0.9,
        early_stopping=True,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text