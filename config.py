from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)