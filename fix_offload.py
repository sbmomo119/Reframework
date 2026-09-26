"""用函数级替换修 _setup_moe_offload，不依赖字符串匹配。"""
from pathlib import Path
import re, shutil

eng = Path("/home/samuel/Re/reframework/engine/engine.py")
shutil.copy(eng, str(eng) + ".funcbak")
src = eng.read_text()

# 找 _setup_moe_offload 函数头
m_start = re.search(r'^(    def _setup_moe_offload\(self.*?\n)', src, re.MULTILINE)
if not m_start:
    print("❌ 没找到 _setup_moe_offload")
    raise SystemExit(1)

# 找下一个同级 def（缩进 4 空格的 def）
rest = src[m_start.end():]
m_next = re.search(r'^(    def \w+\(self)', rest, re.MULTILINE)
if m_next:
    end_pos = m_start.end() + m_next.start()
else:
    end_pos = len(src)

new_func = '''    def _setup_moe_offload(self) -> None:
        """Stage MoE experts into a pinned-host LRU cache (per MoE layer)."""
        self.moe_offload = []
        mc = self.model.cfg
        # mc.has_moe 在部分 ModelConfig 上可能是 None；fallback 检查第一层 mlp
        has_moe = getattr(mc, "has_moe", None)
        if has_moe is None:
            first_mlp = getattr(self.model.layers[0], "mlp", None)
            has_moe = hasattr(first_mlp, "build_offload_cache")
        if not has_moe or not self.ecfg.moe_offload:
            return
        from reframework import env  # local: keep engine import light

        cap = self.ecfg.moe_cache_size
        if cap is None:
            cap = env.get_moe_cache_size()
        if cap is None or cap <= 0:
            cap = min(getattr(mc, "num_experts", 8) or 8, 8)
        for layer in self.model.layers:
            mlp = getattr(layer, "mlp", None)
            if hasattr(mlp, "build_offload_cache"):
                self.moe_offload.append(mlp.build_offload_cache(self.device, cap))
        if self.moe_offload:
            logger.info(
                "moe offload on: %d layer(s), lru_capacity=%d expert(s)",
                len(self.moe_offload), cap,
            )

'''

src = src[:m_start.start()] + new_func + src[end_pos:]
eng.write_text(src)
print("✅ _setup_moe_offload 已替换")
