import os
import sys
from pathlib import Path

# Configure single-thread PyTorch CPU footprint for low-memory container environments (Render 512MB limit)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import uvicorn
from api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[HealRAG] Launching Uvicorn server on 0.0.0.0:{port} (PORT env: {os.environ.get('PORT')})...")
    uvicorn.run(app, host="0.0.0.0", port=port)
