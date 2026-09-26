"""litemoe — Lite MoE expert-offload runtime.

Modules
-------
config          unified config (model path, cache size, strategy)
interface       the four core contracts: ExpertStore / Cache / Predictor / Scheduler
quantization    GGUF int4 dequantization + safetensors int4 packing (QuantScheme)
model           GGUF / safetensors loaders (expert separation) + Qwen3 MoE model
store           ExpertStore backends (GGUFExpertStore / SafetensorsExpertStore)
cache           LRU baseline + PDE strategy + metrics + activation log
predictor       n-gram routing predictor (for PDE lookahead)
scheduler       bandwidth-aware per-step fetch/evict policy
cli             `litemoe run --config xxx`
"""

from litemoe.config import LitemoeConfig
from litemoe.quantization import QuantScheme, Int4, F16
from litemoe.model.loader import GGUFLoader, ModelMeta
from litemoe.model.safetensors_loader import SafetensorsLoader
from litemoe.store import GGUFExpertStore, SafetensorsExpertStore, make_store

__all__ = [
    "LitemoeConfig",
    "QuantScheme",
    "Int4",
    "F16",
    "GGUFLoader",
    "ModelMeta",
    "SafetensorsLoader",
    "GGUFExpertStore",
    "SafetensorsExpertStore",
    "make_store",
]

__version__ = "0.1.0"
