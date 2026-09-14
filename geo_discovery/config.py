"""Model registry, pricing, task-model defaults, paths."""
import os
from pathlib import Path

# ---- credentials: environment only, never committed ----
# Copy .env.example to .env and fill it in. .env is gitignored.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        raise RuntimeError(
            f"Missing required environment variable {name}.\n"
            f"Copy .env.example to .env and fill it in, or export it in your shell.")
    return v

GCP_PROJECT = _env("GCP_PROJECT_ID", required=True)
GCP_REGION = _env("GOOGLE_CLOUD_REGION", "global")
if _env("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _env("GOOGLE_APPLICATION_CREDENTIALS")

os.environ["AZURE_API_KEY"] = _env("AZURE_API_KEY", required=True)
os.environ["AZURE_API_BASE"] = _env("AZURE_API_BASE", required=True)
os.environ["AZURE_API_VERSION"] = _env("AZURE_API_VERSION", "2024-08-01-preview")

# ---- MODEL REGISTRY (provider routing + capabilities) ----
# In MODEL_REGISTRY, tag models with temperature + reasoning constraints:
MODEL_REGISTRY = {
    "gemini-2.5-flash": {"provider": "vertex", "supports": ["text","vision","video","tools"],
                         "temp_policy": "free", "thinking": True, "min_think": 0},
    "gemini-2.5-pro":   {"provider": "vertex", "supports": ["text","vision","video","tools"],
                         "temp_policy": "free", "thinking": True, "min_think": 128},
    "azure/gpt-5-mini": {"provider": "litellm", "supports": ["text","tools"],
                         "temp_policy": "fixed_1", "reasoning": True},
    "azure/gpt-4o":     {"provider": "litellm", "supports": ["text","vision","tools"],
                         "temp_policy": "free"},
    "claude-opus-4":    {"provider": "litellm", "supports": ["text","vision"],
                         "temp_policy": "free"},
    "deepseek-chat":    {"provider": "litellm", "supports": ["text"], "temp_policy": "free"},
}


# ---- PRICING ($ per token) — ILLUSTRATIVE, replace with live rates ----
PRICING = {
    "gemini-2.5-flash": (0.30/1e6, 2.50/1e6),
    "gemini-2.5-pro":   (1.25/1e6, 10.00/1e6),
    "azure/gpt-5-mini": (0.25/1e6, 2.00/1e6),
    "azure/gpt-4o":     (2.50/1e6, 10.0/1e6),
    "claude-opus-4":    (15.0/1e6, 75.0/1e6),
    "deepseek-chat":    (0.14/1e6, 0.28/1e6),
}

# ---- DEFAULT per-task model choices (front-end overrides these) ----
DEFAULT_TASK_MODELS = {
    "extraction": "gemini-2.5-pro",     # doc/positioning/claims
    "vision":     "gemini-2.5-flash",   # ads (later)
    "judge":      "claude-opus-4",      # different family (later modules)
    "query_mining":"gemini-2.5-flash",
}

# ---- Storage ----
STORE_ROOT = Path(os.environ.get("GEO_STORE", "geo_projects"))
STORE_ROOT.mkdir(exist_ok=True)