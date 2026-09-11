import os
import json

# Path to attention files
ATTENTION_DIR = os.path.join(os.path.dirname(__file__), "outputs", "attention_files")

def summarize_attention():
    if not os.path.exists(ATTENTION_DIR):
        print(f"No attention files found in {ATTENTION_DIR}.")
        return

    files = [f for f in os.listdir(ATTENTION_DIR) if f.endswith(".json")]
    if not files:
        print("No JSON attention files found.")
        return

    print(f"Found {len(files)} attention files:\n")
    for fname in files:
        path = os.path.join(ATTENTION_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            num_heads = len(data.get("attentions", []))  # assuming 'attentions' key
            print(f"{fname}: {num_heads} attention heads")
        except Exception as e:
            print(f"{fname}: Error reading file - {e}")

if __name__ == "__main__":
    summarize_attention()
