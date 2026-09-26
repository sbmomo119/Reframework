import torch
from pathlib import Path
from reframework.checkpoint.loader import load_config, build_model, load_weights

ckpt = '/home/samuel/huihui-ai--Huihui-MoE-1.2B-A0.6B/snapshots/master'

def gpu_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024**2
    return 0

cfg_json = load_config(Path(ckpt))
m = build_model(cfg_json, dtype=torch.float32, device='cpu')
print(f"[1] build_model CPU: GPU={gpu_mb():.1f} MB")

sd = load_weights(Path(ckpt))
m.load_state_dict(sd)
print(f"[2] load_weights CPU: GPU={gpu_mb():.1f} MB")

# 对**所有** MoE 层 offload
offloaded = 0
for layer in m.layers:
    mlp = getattr(layer, "mlp", None)
    if mlp is not None and hasattr(mlp, "build_offload_cache"):
        mlp.build_offload_cache(torch.device('cuda:0'), lru_capacity=1)
        offloaded += 1
print(f"[3] offloaded {offloaded} layers; GPU={gpu_mb():.1f} MB")

m.to('cuda:0')
print(f"[4] after .to(cuda): GPU={gpu_mb():.1f} MB")

# 检查所有层专家是否为空
for layer in m.layers[:3]:
    mlp = layer.mlp
    print(f"    layer mlp.experts_w1 len = {len(mlp.experts_w1)}")
