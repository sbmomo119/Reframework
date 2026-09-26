"""修复 reframework 的两个已知 bug：
1. _setup_moe_offload 在 self.model.to 之后调用（顺序错）
2. _setup_moe_offload 的 has_moe fallback（mc.has_moe 是 None）
3. build_offload_cache 释放专家的方式（ParameterList = None 无效）
"""
from pathlib import Path
import shutil

ROOT = Path("/home/samuel/Re/reframework")
LOG = []

# ============ 补丁 1 + 2：engine.py ============
eng = ROOT / "engine" / "engine.py"
shutil.copy(eng, str(eng) + ".fixbak")
lines = eng.read_text().splitlines(keepends=True)

# 1a. 交换 _setup_moe_offload 和 self.model.to 的顺序
to_idx = None
setup_idx = None
for i, l in enumerate(lines):
    if "self.model.to(self.device)" in l and to_idx is None:
        to_idx = i
    if "self._setup_moe_offload()" in l and setup_idx is None:
        setup_idx = i

if to_idx is not None and setup_idx is not None:
    if setup_idx > to_idx:
        # setup 在后面：把 setup 移到 to 之前
        setup_line = lines.pop(setup_idx)
        lines.insert(to_idx, setup_line)
        LOG.append("engine.py: _setup_moe_offload 移到 self.model.to 之前")
    else:
        LOG.append("engine.py: 顺序已经正确")

# 1b. 修 _setup_moe_offload 的 has_moe fallback
src = "".join(lines)
old = """        self.moe_offload = []
        mc = self.model.cfg
        if not getattr(mc, "has_moe", False) or not self.ecfg.moe_offload:
            return"""

new = """        self.moe_offload = []
        mc = self.model.cfg
        # mc.has_moe 在部分 ModelConfig 上可能是 None；fallback 到检查第一层 mlp
        has_moe = getattr(mc, "has_moe", None)
        if has_moe is None:
            first_mlp = getattr(self.model.layers[0], "mlp", None)
            has_moe = hasattr(first_mlp, "build_offload_cache")
        if not has_moe or not self.ecfg.moe_offload:
            return"""

if old in src:
    src = src.replace(old, new)
    LOG.append("engine.py: _setup_moe_offload has_moe fallback 已加")
else:
    LOG.append("engine.py: WARN has_moe fallback 未匹配")

# 1c. 默认 lru_capacity 用 ecfg.moe_cache_size，不强制 8
old = """        cap = self.ecfg.moe_cache_size
        if cap is None:
            cap = env.get_moe_cache_size()
        if cap is None or cap <= 0:
            cap = min(mc.num_experts, 8)  # default window: keep top-8 experts"""

new = """        cap = self.ecfg.moe_cache_size
        if cap is None:
            cap = env.get_moe_cache_size()
        if cap is None or cap <= 0:
            cap = min(getattr(mc, "num_experts", 8) or 8, 8)"""

if old in src:
    src = src.replace(old, new)
    LOG.append("engine.py: lru_capacity fallback 已修")

eng.write_text(src)

# ============ 补丁 3：moe_layer.py ============
ml = ROOT / "moe" / "moe_layer.py"
shutil.copy(ml, str(ml) + ".fixbak")
ml_lines = ml.read_text().splitlines(keepends=True)

start = None
end = None
for i, l in enumerate(ml_lines):
    if "self._cache.load(experts)" in l and start is None:
        start = i
    if start is not None and "return self._cache" in l and end is None:
        end = i
        break

if start is not None and end is not None:
    new_block = [
        "        self._cache.load(experts)\n",
        "        # 彻底释放原专家 ParameterList：把 ParameterList 替换成空的，\n",
        "        # nn.Module 的 _parameters 才会解引用旧 Parameter\n",
        "        self.experts_w1 = torch.nn.ParameterList()\n",
        "        self.experts_w2 = torch.nn.ParameterList()\n",
        "        if torch.cuda.is_available():\n",
        "            torch.cuda.empty_cache()\n",
        "        return self._cache\n",
    ]
    ml_lines[start:end+1] = new_block
    ml.write_text("".join(ml_lines))
    LOG.append(f"moe_layer.py: build_offload_cache 释放补丁 {start+1}-{end+1}")
else:
    LOG.append("moe_layer.py: WARN build_offload_cache 未匹配")

print("\n".join(LOG))
