"""
Stateless LLM demo


Shows that a raw LLM call has NO memory between invocations.
Each call is a fresh context - previous messages are discarded.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


MODEL = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)


def ask(messages: list[dict]) -> str:
    response = llm.invoke(messages)
    return response.content


def main() -> None:
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        # Every call gets only the current message - no conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
        reply = ask(messages)
        print(f"LLM: {reply}\n")


if __name__ == "__main__":
    main()


"""
Usage:
$ python 1_llms_are_stateless.py

Questions to ask the LLM:
1. Who is virat kohli?
2. When was he born?
"""
