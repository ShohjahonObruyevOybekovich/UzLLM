from config import model, tokenizer
import torch

def ask_model(prompt, max_tokens=512):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

    # Faqat yangi qismni olish uchun tokenizer boshi + oxirigacha decode qilinadi
    full_output = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Optionally, extract only the answer after [/INST]
    # Bu garantiyalanmagan, lekin ko‘proq ishonchli
    if "[/INST]" in full_output:
        return full_output.split("[/INST]")[-1].strip()
    else:
        return full_output.strip()
