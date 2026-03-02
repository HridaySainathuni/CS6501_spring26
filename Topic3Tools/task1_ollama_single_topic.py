"""Topic 3 Task 1b: Single-topic MMLU evaluation via Ollama.
Usage: python task1_ollama_single_topic.py --topic astronomy
       python task1_ollama_single_topic.py --topic business_ethics
Requires: pip install ollama datasets tqdm; ollama pull llama3.2:1b (or equivalent)
"""
import sys
import os
import re

SUBJECT = "astronomy"

def format_prompt(question, choices):
    prompt = f"{question}\n\n"
    for i, c in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {c}\n"
    prompt += "\nAnswer with one letter only (A, B, C, or D):"
    return prompt

def get_answer_ollama(prompt, model="llama3.2:1b"):
    try:
        import ollama
        r = ollama.generate(model=model, prompt=prompt)
        text = (r.get("response") or "").strip().upper()
        m = re.search(r"\b([ABCD])\b", text)
        return m.group(1) if m else "A"
    except Exception as e:
        print(f"Ollama error: {e}", file=sys.stderr)
        return "A"

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="astronomy", help="MMLU subject")
    p.add_argument("--model", default="llama3.2:1b", help="Ollama model name")
    args = p.parse_args()
    topic = args.topic

    from datasets import load_dataset
    from tqdm import tqdm

    try:
        dataset = load_dataset("cais/mmlu", topic, split="test")
    except Exception as e:
        print(f"Error loading {topic}: {e}", file=sys.stderr)
        sys.exit(1)

    correct = total = 0
    for ex in tqdm(dataset, desc=topic):
        prompt = format_prompt(ex["question"], ex["choices"])
        pred = get_answer_ollama(prompt, args.model)
        ans = "ABCD"[ex["answer"]]
        if pred == ans:
            correct += 1
        total += 1

    print(f"Topic: {topic}  Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

if __name__ == "__main__":
    main()
