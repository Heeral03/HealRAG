import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import uvicorn
from api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[HealRAG] Launching Uvicorn server on 0.0.0.0:{port} (PORT env: {os.environ.get('PORT')})...")
    uvicorn.run(app, host="0.0.0.0", port=port)
