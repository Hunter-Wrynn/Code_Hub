import os
os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.cache"))

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(model)