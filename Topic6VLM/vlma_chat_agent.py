"""Topic 6 Exercise 1: Vision-Language LangGraph chat agent.
Multi-turn conversation about an uploaded image. Uses Ollama with LLaVA or moondream.
Run: ollama pull llava  (then)  python vlma_chat_agent.py [image_path] [--model llava|moondream]
"""
import sys
import os

def main():
    try:
        import ollama
    except ImportError:
        print("Install ollama: pip install ollama")
        return
    args = [a for a in sys.argv[1:] if a and not a.startswith("-")]
    model = "llava"
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
    image_path = args[0] if args else None
    if not image_path or not os.path.isfile(image_path):
        print("Usage: python vlma_chat_agent.py <image_path> [--model llava|moondream]")
        print("Example: python vlma_chat_agent.py photo.jpg --model moondream")
        return
    messages = []
    print("Vision-language chat. Describe or ask about the image. Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        # Each turn: send conversation history + current question; image only in first user msg
        if not messages:
            messages.append({
                "role": "user",
                "content": user_input,
                "images": [image_path],
            })
        else:
            messages.append({"role": "user", "content": user_input})
        try:
            r = ollama.chat(model=model, messages=messages)
            raw = r.get("message") or {}
            content = (raw.get("content") or "").strip()
            print("Assistant:", content or "(no response)")
            messages.append({"role": "assistant", "content": content or ""})
        except Exception as e:
            err_str = str(e)
            if model == "llava" and ("500" in err_str or "unexpectedly stopped" in err_str):
                try:
                    r = ollama.chat(model="moondream", messages=messages)
                    raw = r.get("message") or {}
                    content = (raw.get("content") or "").strip()
                    print("Assistant:", content or "(no response)")
                    messages.append({"role": "assistant", "content": content or ""})
                    model = "moondream"
                    continue
                except Exception:
                    pass
            print("Error:", e)
            messages.pop()  # remove failed user msg
    print("Goodbye.")

if __name__ == "__main__":
    main()
