import os
import json
import torch
import argparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- CLI Argument Parsing ---
parser = argparse.ArgumentParser(description="VizFold API Bridge")
parser.add_argument(
    "--dir", 
    type=str, 
    default="test_output", 
    help="Path to the inference output directory (default: test_output)"
)
# We use parse_known_args so it doesn't crash if run with external uvicorn flags
args, _ = parser.parse_known_args()

# Lock down the base directory based on CLI input
BASE_DIR = os.path.abspath(args.dir)

app = FastAPI(title="VizFold API Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_join(directory, *paths):
    """Ensures requested files cannot escape the BASE_DIR."""
    requested_path = os.path.abspath(os.path.join(directory, *paths))
    if not requested_path.startswith(BASE_DIR):
        raise HTTPException(status_code=403, detail="Path traversal attempt blocked.")
    return requested_path

@app.get("/")
def read_root():
    return {"status": "VizFold API is live", "directory": BASE_DIR}

@app.get("/meta")
def get_meta():
    meta_path = safe_join(BASE_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="meta.json not found")
    with open(meta_path, "r") as f:
        return json.load(f)

@app.get("/structure")
def get_structure():
    pdb_path = safe_join(BASE_DIR, "structure", "predicted.pdb")
    if not os.path.exists(pdb_path):
        raise HTTPException(status_code=404, detail="PDB structure not found")
    return FileResponse(pdb_path, media_type="text/plain")

@app.get("/tensor/{category}/{filename}")
def get_tensor(category: str, filename: str):
    # 1. Explicitly block path traversal characters in the filename or category
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid characters in filename")
    if ".." in category or "/" in category or "\\" in category:
        raise HTTPException(status_code=400, detail="Invalid characters in category")

    # 2. Security check: only allow reading from specific trace directories
    if category not in ["attention", "activations", "structure_module"]:
        raise HTTPException(status_code=400, detail="Invalid trace category")
        
    filepath = safe_join(BASE_DIR, "trace", category, f"{filename}.pt")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Tensor {filename}.pt not found")
        
    try:
        tensor = torch.load(filepath, weights_only=True)
        tensor = tensor.float().detach().cpu()
        
        # 1. Average Trunk Evolution (s_z) across the 128 hidden channels
        if category == "activations" and "s_z" in filename:
            tensor = tensor.mean(dim=-1)
            
        # 2. Average ESM-2 Attention across all Attention Heads
        elif category == "attention" and "layer" in filename:
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0) 
            tensor = tensor.mean(dim=0)

        tensor_list = tensor.numpy().tolist()
        
        return {
            "name": filename,
            "shape": list(tensor.shape),
            "data": tensor_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing tensor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)