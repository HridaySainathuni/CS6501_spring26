"""Setup Verification Script"""
import os
import sys
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
try:
    from load_secrets import load_secrets
    load_secrets()
except Exception:
    pass
import subprocess
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"[OK] {package_name} is installed")
        return True
    except ImportError:
        print(f"[FAIL] {package_name} is NOT installed")
        return False

def check_huggingface_auth():
    try:
        # Prefer env (HF_TOKEN or HUGGING_FACE_HUB_TOKEN), then get_token() (huggingface_hub >= 0.20)
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except ImportError:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
        if token:
            print("[OK] Hugging Face is authenticated")
            return True
        else:
            print("[FAIL] Hugging Face is NOT authenticated")
            print("  Run: huggingface-cli login")
            return False
    except Exception as e:
        print(f"[FAIL] Error checking Hugging Face authentication: {e}")
        return False

def main():
    print("="*70)
    print("Setup Verification")
    print("="*70)
    print()
    
    required_packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("tqdm", "tqdm"),
        ("huggingface_hub", "huggingface_hub"),
        ("bitsandbytes", "bitsandbytes"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("numpy", "numpy"),
        ("psutil", "psutil"),
    ]
    
    print("Checking required packages...")
    print("-"*70)
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    print("Checking Hugging Face authentication...")
    print("-"*70)
    hf_authenticated = check_huggingface_auth()
    
    print()
    print("="*70)
    if all_installed and hf_authenticated:
        print("[OK] Setup is complete! You're ready to run the evaluation.")
    else:
        print("[FAIL] Setup is incomplete. Please install missing packages and authenticate.")
        if not all_installed:
            print("\nTo install missing packages, run:")
            print("  pip install -r requirements.txt")
    print("="*70)

if __name__ == "__main__":
    main()

