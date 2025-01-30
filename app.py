import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузите модель и токенизатор
model = AutoModelForCausalLM.from_pretrained("./volga_v0.3")
tokenizer = AutoTokenizer.from_pretrained("./volga_v0.3")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Создайте интерфейс
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="Volga Model",
    description="Модель для генерации текста на русском языке."
)

# Запустите приложение
iface.launch()