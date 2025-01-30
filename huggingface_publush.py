from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузите вашу обученную модель и токенизатор
model = AutoModelForCausalLM.from_pretrained("./volga_v0.3")
tokenizer = AutoTokenizer.from_pretrained("./volga_v0.3")

# Загрузите их на Hugging Face Hub
model.push_to_hub("kusiq/volga-v0.3-KROT")
tokenizer.push_to_hub("kusiq/volga-v0.3-KROT")