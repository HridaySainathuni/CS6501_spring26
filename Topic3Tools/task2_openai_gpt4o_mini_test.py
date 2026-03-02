"""Topic 3 Task 2: OpenAI GPT-4o Mini test.
Set OPENAI_API_KEY in environment or in repo root secrets.json (see secrets.json.example).
"""
import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from load_secrets import load_secrets
    load_secrets()
except Exception:
    pass

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Set it in environment or use Colab Secrets.")
        return

    # client = OpenAI() uses OPENAI_API_KEY from env by default
    from openai import OpenAI
    client = OpenAI()  # Uses os.getenv("OPENAI_API_KEY")

    # client.chat.completions.create() sends a request to OpenAI API;
    # messages is the conversation, model selects the model, max_tokens caps response length
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say: Working!"}],
        max_tokens=5,
    )
    text = response.choices[0].message.content
    print("Response:", text)
    if "Working" in (text or ""):
        print("OpenAI GPT-4o Mini test passed.")

if __name__ == "__main__":
    main()
