"""Topic 3 Task 1a: Single-topic MMLU evaluation (Hugging Face).
Run on one MMLU subject. Use with task1_mmlu_topic2.py for sequential/parallel timing.
Usage: python task1_mmlu_single_topic.py --topic astronomy
       python task1_mmlu_single_topic.py --topic business_ethics
"""
import sys
import os
# Allow importing from parent repo root if running from Topic3Tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from tqdm import tqdm

# Use Topic 1 eval logic; minimal deps for single-topic
SUBJECT = os.environ.get("MMLU_TOPIC", "astronomy")

def format_prompt(question, choices):
    prompt = f"{question}\n\n"
    for i, c in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {c}\n"
    prompt += "\nAnswer:"
    return prompt

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default=os.environ.get("MMLU_TOPIC", "astronomy"), help="MMLU subject")
    args = p.parse_args()
    topic = args.topic

    try:
        dataset = load_dataset("cais/mmlu", topic, split="test")
    except Exception as e:
        print(f"Error loading {topic}: {e}", file=sys.stderr)
        sys.exit(1)

    # Use Hugging Face pipeline if available
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.backends.mps.is_available():
            device = "mps"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"HF model load failed (run from repo with transformers installed): {e}", file=sys.stderr)
        sys.exit(1)

    correct = total = 0
    for ex in tqdm(dataset, desc=topic):
        prompt = format_prompt(ex["question"], ex["choices"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()[:1].upper()
        if pred not in "ABCD":
            pred = "A"
        ans = "ABCD"[ex["answer"]]
        if pred == ans:
            correct += 1
        total += 1

    print(f"Topic: {topic}  Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

if __name__ == "__main__":
    main()
