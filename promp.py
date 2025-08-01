from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def build_prompt(history, user_input, system_prompt):
    if not history:
        # Birinchi prompt
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_input} [/INST]"
    else:
        # Oldingi tarix asosida prompt yaratish
        prompt = ""
        for turn in history:
            prompt += f"<s>[INST] {turn['user']} [/INST] {turn['assistant']} </s>"
        prompt += f"<s>[INST] {user_input} [/INST]"
    return prompt

