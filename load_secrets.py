"""Load API keys from secrets.json into environment. Keys stay out of code and .gitignore."""
import os
import sys

def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    return d

def load_secrets(path=None):
    if path is None:
        path = os.path.join(_repo_root(), "secrets.json")
    if not os.path.isfile(path):
        return False
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                os.environ[key] = value.strip()
        # So Hugging Face sees the token (some code checks one or the other)
        if "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        elif "HUGGING_FACE_HUB_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
        return True
    except Exception:
        return False

if __name__ == "__main__":
    if load_secrets():
        print("Loaded keys from secrets.json into environment.")
    else:
        print("No secrets.json found or error reading it.")
