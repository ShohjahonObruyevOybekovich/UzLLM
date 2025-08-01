from ask_model import ask_model
from promp import build_prompt

def run_chat():
    history = []
    system_prompt = "Sen aqlli yordamchisan va o‘zbek tilida foydalanuvchiga do‘stona javob berasan."

    while True:
        try:
            user_input = input("👤 Siz: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            prompt = build_prompt(history, user_input, system_prompt)
            response = ask_model(prompt)

            print(f"🤖 Bot: {response}\n")

            history.append({
                "user": user_input,
                "assistant": response
            })

        except KeyboardInterrupt:
            print("\n⛔ Chat tugadi.")
            break

if __name__ == "__main__":
    run_chat()
