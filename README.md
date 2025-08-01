# 🇺🇿 uzLLM — Train Your Own Uzbek Large Language Model  
> 🧠 Fine-tune, instruct, and deploy Uzbek AI assistants with HuggingFace Transformers.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" width="120" />
  <br/><br/>
  <strong>Uzbek-first LLM training pipeline</strong><br/>
  Built on 🤗 Transformers · ⚡ PyTorch · 🧪 Prompt Engineering · 🚀 Accelerate
</div>

---

## 📌 What is uzLLM?

`uzLLM` is a lightweight but powerful training framework for **fine-tuning open-source LLMs (like Mistral, LLaMA, Falcon)** on **Uzbek datasets** — enabling full control over instruction-following AI, local assistant models, or even Uzbek GPT-style chatbots.

---

## ✨ Features

✅ Fine-tune HuggingFace LLMs on Uzbek text  
✅ Instruction-style prompt formatting (OpenAssistant style)  
✅ Supports Mistral / LLaMA / Falcon / Qwen (any causal LM)  
✅ Multi-GPU or Colab-compatible  
✅ Clean conversational chat history handling  
✅ Easy inference API via CLI or Streamlit (optional)

---

## 🛠️ Quickstart

```bash
git clone https://github.com/your-username/uzllm.git
cd uzllm
pip install -r requirements.txt
Prepare your dataset in .jsonl format:

json
Copy
Edit
{"instruction": "Toshkent qaerda joylashgan?", "response": "Toshkent Oʻzbekiston poytaxti."}
Start fine-tuning:

bash
Copy
Edit
python train.py \
  --model_name_or_path NousResearch/Nous-Hermes-2-Mistral-7B-DPO \
  --train_file data/uzbek-instructions.jsonl \
  --output_dir models/uzllm-mistral \
  --per_device_train_batch_size 2 \
  --num_train_epochs 3 \
  --fp16
🧪 Example Prompt Format
Your data should follow the [INST] style used in LLaMA/Mistral:

text
Copy
Edit
<s>[INST] <<SYS>>
Siz foydalanuvchiga o‘zbek tilida aqlli, aniq va foydali javoblar berasiz.
<</SYS>>

Salom! Bugun ob-havo qanday?
[/INST]
📦 Inference
python
Copy
Edit
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("models/uzllm-mistral")
model = AutoModelForCausalLM.from_pretrained("models/uzllm-mistral")

prompt = "<s>[INST] Salom, sen kimsan? [/INST]"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
🧰 Tech Stack
🤗 Transformers

🧨 Datasets

🧪 Accelerate

🧠 PyTorch

🗂️ Project Structure
bash
Copy
Edit
uzllm/
├── data/                  # Uzbek training data (jsonl)
├── models/                # Output fine-tuned models
├── train.py               # Training script
├── infer.py               # Inference demo
├── prompts/               # Prompt templates
└── README.md
🙌 Contribute Uzbek AI 🇺🇿
We're just getting started. If you're working with Uzbek NLP or LLMs — join us!

🧠 Add new Uzbek datasets

🔥 Share fine-tuned checkpoints

💬 Improve system prompts

🤖 Add inference UI (Streamlit / Telegram bot)

🏁 Coming Soon
 Streamlit chat demo

 LoRA-based training

 HuggingFace Hub auto-push

 Uzbek QA dataset release

📜 License
MIT — do whatever you want, just give credit.
Made with ❤️ for O'zbek tilida AI rivoji uchun.

✨ "Why wait for OpenAI to support Uzbek? Train your own."

yaml
Copy
Edit

---

## 💡 Next Steps I Can Help With

- `train.py` and `infer.py` templates
- Streamlit UI for chatting
- HuggingFace Hub push setup (`transformers-cli`)
- Dataset format validators

Let me know if you want it zipped or deployed to GitHub — I’ll make it repo-ready 🚀








Ask ChatGPT
