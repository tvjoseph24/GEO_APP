"""Model registry, pricing, task-model defaults, paths."""
import os
from pathlib import Path

# ---- Vertex / GCP ----
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      "/Users/thomasjoseph/Desktop/JMJTL/Projects/Gen_ai/key1.json")
GCP_PROJECT = os.environ.get("GCP_PROJECT_ID", "spring-outlet-434817-q4")
GCP_REGION  = os.environ.get("GOOGLE_CLOUD_REGION", "global")

# ---- LiteLLM / Azure (for non-Gemini) ----
os.environ.setdefault("AZURE_API_KEY",     os.environ.get("AZURE_API_KEY", "sk-cWKLpS1aqz2lJ1x6vl7R0w"))
os.environ.setdefault("AZURE_API_BASE",    os.environ.get("AZURE_API_BASE", "https://hogarth.litellm-prod.ai"))
os.environ.setdefault("AZURE_API_VERSION", "2024-08-01-preview")

# ---- MODEL REGISTRY (provider routing + capabilities) ----
# In MODEL_REGISTRY, tag models with temperature + reasoning constraints:
MODEL_REGISTRY = {
    "gemini-2.5-flash": {"provider": "vertex",  "supports": ["text","vision","video","tools"],
                         "temp_policy": "free"},
    "gemini-2.5-pro":   {"provider": "vertex",  "supports": ["text","vision","video","tools"],
                         "temp_policy": "free"},
    "azure/gpt-5-mini": {"provider": "litellm", "supports": ["text","tools"],
                         "temp_policy": "fixed_1", "reasoning": True},   # ← GPT-5 constraints
    "azure/gpt-4o":     {"provider": "litellm", "supports": ["text","vision","tools"],
                         "temp_policy": "free"},
    "claude-opus-4":    {"provider": "litellm", "supports": ["text","vision"],
                         "temp_policy": "free"},
    "deepseek-chat":    {"provider": "litellm", "supports": ["text"],
                         "temp_policy": "free"},
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