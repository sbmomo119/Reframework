# CoPlacE 实施计划 — 消费级双卡 + SSD 四层 MoE 推理优化框架

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** 在 `reframework`（本地 FreeToken 同构框架）上实现 CoPlacE：双 GPU 非对称专家缓存、SSD 第四层、路由结构感知替换（RSE）、PTR 尾延迟自适应预取、跨设备路由微调，全部 CPU 可测、生产目标 2×RTX 3080Ti。

**Architecture:** 新增 `reframework/coplace/` 包（配置、设备穿梭、放置规划、SSD 层、延迟模型、预取控制器、RSE 驱逐），扩展现有 `ExpertOffloadCache`/`FusedMoE`/`ReEngine`（保持现有 RE_MOE_* 路径为默认，CoPlacE 通过 RE_COPLAC_* 环境变量开关）。所有跨设备/SSD I/O 走可插拔接口（`DeviceShuttle`、`SSDBackend`），WSL 开发机用 CPU mock，3080Ti 目标机用 CUDA stream + mmap 实现。

**Tech Stack:** Python 3.13, torch 2.7 (cu118 开发 / cu130 目标), safetensors, numpy; 可选 networkx（有 numpy 回退）; pytest; Makefile 入口 `make test`。

---

## 1. 环境现实核查（实施前必读）

本 WSL 开发机与目标机不同，实施时按此分工：

| 项 | 本 WSL 机（开发/单测） | 目标机（生产/实验） |
|---|---|---|
| GPU | 1× GT 730 (2GB, sm_61, 驱动 475.14) | 2× RTX 3080Ti 12GB (sm_86, 驱动 610.57.04) |
| CUDA | 11.4/11.8 (torch cu118) | 需 13.0+（FreeToken JIT 内核） |
| SSD | 仅虚拟盘 (sdd 1T "Virtual Disk") | SL7000 40Pro 2TB NVMe PCIe4.0 x2（需先稳态化） |
| CPU | 16C | Ryzen 9 3955WX 16C/32T |

**推论：**
- 所有新模块必须**无 GPU 可单测**（沿用 `tests/test_moe_prefetch.py` 的 CPU 模式：`device=cpu`，带宽上限直接设 `_bw_gb_s`）。
- CUDA stream / mmap / 真实 PCIe 行为**只能在目标机验证**；本计划中的 CUDA 实现写成独立小函数（`cuda_shuttle.py`），单测用 mock 覆盖逻辑，目标机跑 `scripts/coplace_hardware_check.py` 冒烟。
- 目标机 FreeToken 本体在本仓库不存在（`/home/samuel` 下无 freetoken 目录）；`reframework` 即"FreeToken 结构"的本地等价物，本计划全部落在 `reframework` 上，文档中 `python/freetoken/...` 的改造点对应关系见 §3。
- litemoe 子项目已有：GGUF/safetensors int4 加载、LRU/PDE 缓存、ngram 预测器、激活日志（`litemoe/data/activation_logs/*.jsonl`，格式 `{"step","layer","experts":[..]}`）——共激活图与路由微调的数据来源。

## 2. 设计原则（来自方案文档的硬约束）

1. **token 级分工，不做 EP All-to-All**：隐藏态 4-8KB ≪ 专家权重几十 MB；现有 `reframework/parallel/ep.py`（NCCL）是数据中心路径，CoPlacE 不复用其通信原语，只复用其 `shard_range` 式的区间切分思想。
2. **SSD 只读 mmap**（`ACCESS_READ`/MAP_PRIVATE），绝不回写；fio 证据：混合读写有效读带宽只有纯读 1.5%。
3. **q*/带宽预算必须扣除 SSD 预取占用**（文档 §4.4/§7.4）：`ExpertOffloadCache.reserve()` 现有 budget = `bw_gb_s × forward_ms × headroom`，新增 in-flight 预取流量扣减。
4. **prefill 阶段暂停/降级 SSD 预取**（整层双缓冲占带宽）；`Prediction.record_step(is_prefill=True)` 已重置历史，预取门控挂在这条线上。
5. **PTR（预取及时到达率）为核心指标**：`PTR(D,l)=P(latency≤W_l)`，SSD 延迟按混合分布建模（快路径 lognormal P50≈101µs/P99≈135µs 占 99.5%；慢路径 1-10ms 占 0.5%，P99.9≈2.8ms）。
6. **降级链**：GPU0 未命中 → GPU1 → DRAM → SSD；每层替换/回退都要保持 `ExpertOffloadCache` 的 stats 语义（`OffloadStats`）。

## 3. 文档改造点 → 本仓库文件映射

| 方案文档中的位置 | 本仓库实际落点 | 动作 |
|---|---|---|
| `python/freetoken/moe/offload_cache.py`（OffloadMoeCache/LRU/q*） | `reframework/moe/offload_cache.py`（`ExpertOffloadCache`+`Prediction`） | 扩展：多池缓存、RSE 驱逐、带宽扣减 |
| `python/freetoken/engine/engine.py`（整层双缓冲/弹性内存） | `reframework/engine/engine.py`（`_setup_moe_offload` L415、`_load_moe_sim_tables` L466） | 扩展：CoPlacE 装配钩子 + `reframework/env.py` 新环境变量 |
| `python/freetoken/layers.py`（MoELayer/OffloadMoELayer） | `reframework/moe/moe_layer.py`（`FusedMoE`，`_forward_streaming`/`_forward_split`） | 扩展：双 GPU token 分流 forward 分支 |
| FreeToken 转换工具链（FTW） | `scripts/compute_moe_sim.py` 同风格新脚本 | 新建 `scripts/coplace_*` |
| FreeToken 要求 CUDA13/驱动 r580+ | 目标机冒烟 | `scripts/coplace_hardware_check.py` |
| Fate 式跨层门控预测器 | `reframework/moe/offload_cache.py::Prediction`（已存在，频率+recency 排名） | 直接复用，RSE/PTR 都消费它的输出 |
| LiteMoE 式放置规划器 | `litemoe/src/litemoe/`（cache/pde.py、predictor/ngram.py、激活日志） | 新建 `reframework/coplace/planner.py`（数据源用 litemoe 激活日志） |
| ReMoE 损失复用 | — | 新建 `scripts/coplace_router_finetune.py`（自包含实现 Trust-KL/Smoothness/CrossDevice，不引外部仓库依赖） |

## 4. 新增模块设计（`reframework/coplace/`）

包结构（每个文件都小、CPU 可测）：

```
reframework/coplace/
  __init__.py            # 导出 CoPlacEConfig, build_shuttle, SSDBackend 等
  config.py              # CoPlacEConfig dataclass：设备表、ratio、tier 位宽、开关
  shuttle.py             # DeviceShuttle 抽象 + CpuShuttle(mock) + CudaShuttle(stream)
  placement.py           # PlacementPlanner：频率/共激活/联合 三种策略 -> {expert: tier}
  coactivation.py        # 从 litemoe 激活日志构建共激活矩阵；Louvain 社区（nx 可选，numpy 贪心回退）
  ssd_tier.py            # SSDBackend 抽象 + FileSSDBackend(mmap 只读) + MockSSDBackend
  latency.py             # SSDLatencyModel（混合分布 CDF/采样）、PTR 统计
  prefetch.py            # AdaptivePrefetchDepth + PTRPrefetchController（消费 Prediction 输出）
  rse.py                 # RoutingAwareEviction：消费 sim 表 + 共激活分数的 victim 打分
  router_loss.py         # CrossDeviceRouterLoss（Gumbel-Softmax 设备熵惩罚，纯 torch CPU 可测）
```

### 4.1 与现有组件的集成点（最小侵入）

| 现有符号（行号已核对） | 集成方式 |
|---|---|
| `ExpertOffloadCache`（offload_cache.py:211，dataclass，字段含 `_host`/`_lru`/`_bw_gb_s`/`_bw_headroom`/`_next_reserve`/`stats`） | 新增可选字段 `shuttle=None`、`rse=None`、`prefetch_ctrl=None`、`inflight_bytes=0.0`；`reserve()`（:553）budget 计算前扣减 `inflight_bytes`；`_evict()`（:624）先问 `rse.pick_victim(cands)`，无 rse 时走原 LRU 路径 |
| `Prediction`（:105） | 不改逻辑；`prefetch.py` 订阅其 `predict_next()` 输出做 SSD 预取决策 |
| `FusedMoE`（moe_layer.py:57） | 新增 `enable_dual_gpu(shuttle, placement)`：`forward`（:357）路由后按 placement 把 GPU1 专家的 token 分流（hidden 走 shuttle，权重走对端 cache），GPU0 侧合并输出；单卡/CPU 时该分支自动关闭（`shuttle is None`） |
| `ReEngine._setup_moe_offload`（engine.py:415） | 新增 `_setup_coplac()`：读 `env.get_coplac_*`，按 tier 比例给每个 MoE 层建 placement，给 cache 挂 rse/prefetch_ctrl；失败（缺 SSD 路径等）降级为普通 offload 并打日志 |
| `reframework/env.py`（187 行，现有 RE_MOE_* 一组） | 追加 RE_COPLAC_* 一组（见 Task 1） |
| `EngineConfig`（engine.py:45） | 追加 `coplac: Optional[dict] = None` 透传（chat/server CLI 不强制暴露，先走 env） |
| `reframework/parallel/ep.py::shard_range`（:45） | 只读复用：放置区间切分用它的语义，不改文件 |

> **2026-09-26 `reserve()` 带宽语义审计（本机冒烟已验证，Task 4 前置结论）：**
> - `_bw_gb_s`（L249）是**一次性测量常量**：`attach_predictor()`（L368）调 `measure_bandwidth()`（L420-448）做一次 host→VRAM 拷贝基准并存储，运行时不再复测；CPU 机恒为 None。
> - **但 `reserve()`（:553）已有 per-call `bandwidth_gb_s` keyword-only 覆盖参数**（L558/L585），优先级高于存储常量 → 三方竞争"运行时可更新"的**前置条件已满足**，无需先把 `_bw_gb_s` 改成可变，零结构改动即可挂载。
> - budget 是**字节量**：L590 `budget = bw × 1e9 × (forward_ms/1000) × _bw_headroom` → `inflight_bytes` 扣减 = 直接相减 `budget = max(0, budget − inflight_bytes)`，与论文公式 `(bw − bw_A) × t` 恒等；A（token 传输）与 C（SSD 预取）可共用一个扣减槽。
> - `_bw_headroom` 默认 0.9，`attach_predictor()` 时从 `RE_MOE_PREDICT_HEADROOM` 一次性设置（L367）。
> - **边界情况 L599-600**：首个预测专家即使超 budget 仍会被加载（`if b > budget and n_loaded > 0: break` 只在已加载过 ≥1 个时截断）→ PTR/CDF 约束形式化必须覆盖"一个都放不下"的情形。
> - `Prediction` docstring（L117-118）已预留钩子："a learned router-pattern MLP would replace just `predict_next` and plug in here" → 设备感知预测器按原计划**子类化覆盖 `predict_next`** 接入。
> - 现状代码里流量 C（SSD→DRAM）尚不存在（无 SSD tier）；形式化从三方写起，实现按 A→(A+B)→(A+B+C) 分期，C 随 SSDTierManager 任务落地。

### 4.2 四层 tier 模型

`Tier` 枚举：`TIER_GPU0=0, TIER_GPU1=1, TIER_DRAM=2, TIER_SSD=3`。

- 每层 MoE 的放置是一个 `dict[int, Tier]`（expert_id → tier），由 `PlacementPlanner` 生成并缓存到 `ckpt_dir/coplace_placement.json`（与 `moe_expert_sim.pt` 并列，见 `_load_moe_sim_tables` 的加载惯例）。
- 比例默认：GPU0 15%（8-bit 语义，开发期用 fp16 占位）、GPU1 15%、DRAM 40%、SSD 30%（方案 §3.1）。
- **本仓库当前所有专家权重实际都住在 pinned host（`_host` 列表）**，即 DRAM 层；GPU 层 = LRU 窗口内的驻留专家。CoPlacE 的 `ExpertOffloadCache` 扩展保持这个现实：GPU0 池 = 现有 `_lru`，GPU1 池 = 第二个 `OrderedDict`（`_lru1`），DRAM = `_host`（int8 可选，复用 `quantize_int8_host`），SSD = `SSDBackend`。这样**单卡 3080Ti 也退化为 GPU0+DRAM+SSD 三层**，双卡时 GPU1 池启用。

### 4.3 DeviceShuttle 契约

```python
class DeviceShuttle(Protocol):
    def send(self, hidden: torch.Tensor, to: str) -> torch.Tensor: ...      # 异步
    def recv(self, ref) -> torch.Tensor: ...                                 # 完成事件后取回
    def inflight_bytes(self) -> float: ...                                   # 供 q* 扣减
    def sync(self) -> None: ...
```

- `CpuShuttle`（mock）：立即返回 clone；`inflight_bytes` 返回 0；用于全部单测。
- `CudaShuttle`：每方向一个 `torch.cuda.Stream` + 事件对；`send` 用 `non_blocking=True` 拷贝并记录字节数；`inflight_bytes` 累计未 sync 的字节。**PCIe 双向竞争假设：GPU0↔GPU1 共享一条 x16 链路（无 NVLink），send/recv 串行记账**——这是文档"带宽竞争性使用"叙事的核心假设，冒烟脚本要实测验证。

### 4.4 SSDBackend 契约

```python
class SSDBackend(Protocol):
    def contains(self, layer: int, eid: int) -> bool: ...
    def read(self, layer: int, eid: int) -> tuple[dict[str, torch.Tensor], float]:  # (weights, latency_s)
    def write_layout(self, layer: int, weights: dict, offset: int) -> None: ...      # 布局期
    def latency_model(self) -> "SSDLatencyModel": ...
```

- `FileSSDBackend`：单文件 `<ckpt>/coplace_ssd/experts.bin`，专家按共激活社区连续排布（offset 表写 `layout.json`）；读路径 `mmap` 只读切片 → `torch.frombuffer(..., dtype=uint8)` → dequant；`latency_s` 用 `perf_counter` 实测（目标机）或 `SSDLatencyModel.sample()`（mock）。
- `MockSSDBackend`：内存 dict + 从 `SSDLatencyModel` 采样延迟，单测用。
- **layout 生成**：`coactivation.py` 从 litemoe 激活日志（`{"step","layer","experts":[..]}`）算 per-layer 共激活计数矩阵；社区划分优先 `nx.community.louvain_communities`，import 失败退化为按度排序贪心分块（保证无依赖也能跑）。

### 4.5 RSE（RoutingAwareEviction）

输入：`sim_table[E,E]`（已有，`set_sim_table`）、共激活矩阵、`Prediction.predict_next()` 的 top-k 集合。
victim 打分（在 `_evict` 的候选集上）：
```
score(e) = inf                                    if e ∈ current_step_set
         = 0                                      if e ∈ next_predict_set   # 软保护，与现有 _next_reserve 语义一致
         = coact(e, layer) + λ·sim_max(e, resident)   # 越低越先驱逐
```
保留现有"hard(current)/soft(next_reserve)/stale"两级保护的调用顺序（`_evict` :624-681 结构不动），RSE 只替换 stale 候选的**排序函数**——这是最小侵入点。

### 4.6 PTR 预取控制器

- `SSDLatencyModel`：方案 §4.4.2 的混合分布（快路径 lognormal μ=log(101µs) σ=0.3 w=0.995；慢路径 N(3ms,2ms²) w=0.005）；提供 `cdf(t)`、`sample()`、`quantile(p)`。**参数默认值可被 `coplace_latency.json` 覆盖**（目标机 fio 实测后回填）。
- `AdaptivePrefetchDepth`：D∈[1,4]，窗口 W_l 用最近 D+1 个 forward 实测耗时和；`ptr = timely/total`（EMA window=100）；升 D 条件 `avg_ptr < 0.99`，降 D 条件 `avg_ptr > 0.995`（滞回 0.005，防抖）。
- `PTRPrefetchController.trigger(layer_idx, next_set)`：`next_set` 来自 `Prediction.predict_next()`；缺 SSD 层且 DRAM 未命中 → 发起 `SSDBackend.read`（mock 下同步；CUDA 机走独立 stream）→ 完成后 `cache.inflight_bytes -= n`；**prefill 批（`is_prefill=True`）直接 return**（文档 §7.5）。
- 与 q* 协同：`reserve()` budget 扣减 = `shuttle.inflight_bytes() + ssd.inflight_bytes()`（方案 §4.4/§7.4 的"扣除预取流量"）。

## 5. 任务分解（TDD，每步可验证）

> 约定：`pytest -q tests/<file>::<test>` 为本计划所有测试的运行方式；提交粒度=每 Task 一次 `git add <files> && git commit`。分支：在 `master` 上直接做（仓库现状 74 staged/20 modified/58 untracked 是既有状态，**不要**替用户整理或提交既有文件；只提交本计划新增/修改的文件）。
> **Task 0 是前置闸门**（真 GPU stream 语义验证）：先跑并存档结果；实测语义与 mock 假设冲突则停下修契约，不继续后续任务。

### Task 0（前置闸门）: 真 GPU stream 语义冒烟 — 先于所有任务执行（1h）

> 理由：最大架构风险是"CUDA stream 异步预取 × q* 带宽预算扣减 × non_blocking 拷贝"——mock 语义若与真路径不符，写完 12 个任务才在目标机上发现，返工成本最高。先在真 GPU 上把这条链路实测并固化契约，后续所有 mock 任务围绕已验证契约设计。

**Files:** Create `scripts/coplace_gpu_smoke.py`、`tests/test_coplac_gpu_smoke.py`；报告存档 `docs/coplace_gpu_smoke_$(date).txt`

脚本验证 4 项（单卡项在本 WSL 机 GT 730 即可跑；双卡项标 `SKIP: 目标机 only`）：
1. `torch.cuda.Stream` + event 对：`non_blocking=True` 的 H2D/D2D 拷贝确实与默认流计算重叠（测 overlap 比例，<50% 则 mock 的"异步"假设需重审）；
2. inflight 记账不变式：已知字节数 send→sync 之间 `inflight_bytes` 单调递减、sync 后归零（Task 5/6 的 `_Accounting` 记账模型以此为准）；
3. reserve() 扣减公式：`budget -= inflight`，inflight=0.5×budget → k 减半（与 Task 6 单测同一公式，保证 mock 测试与真路径扣减一致）；
4. （仅双卡）GPU0↔GPU1 D2D 单向/双向并发吞吐比 → 实测验证"共享 PCIe 链路竞争"假设（R4）。

**Gate:** 单卡项任一失败脚本 exit non-zero → 停下先修 mock 契约再执行 Task 1+；Task 5/6 单测引用此报告为 mock 语义依据。GPU 完全不可用时如实报告（不得伪造），闸门降级为"目标机执行"。

### Task 1: env 开关 + CoPlacEConfig（0.5h）

**Files:** Modify `reframework/env.py`（追加 ~40 行）、Create `reframework/coplace/config.py`、`reframework/coplace/__init__.py`

`env.py` 追加（沿用现有 `_get_bool/_get_int/_get_float` 惯例）：
```python
def get_coplac() -> bool:
    return _get_bool("RE_COPLAC", False)
def get_coplac_devices() -> list[str]:
    return [d.strip() for d in _get_str("RE_COPLAC_DEVICES", "cuda:0,cuda:1").split(",") if d.strip()]
def get_coplac_ratios() -> tuple[float, float, float]:
    """gpu0, gpu1, dram 比例；ssd = 1 - 三者之和（floor 0）"""
    g0, g1, dr = (float(x) for x in _get_str("RE_COPLAC_RATIOS", "0.15,0.15,0.40").split(","))
    return g0, g1, dr
def get_coplac_ssd_path() -> str | None:
    return _get_str("RE_COPLAC_SSD_PATH", "") or None
def get_coplac_ptr_target() -> float:
    return _get_float("RE_COPLAC_PTR_TARGET", 0.99)
def get_coplac_min_experts_per_gpu() -> int:
    """每 GPU 专家槽位下限（防小模型比例切分后槽位过少：Mixtral 8 专家×15%≈1）"""
    return _get_int("RE_COPLAC_MIN_EXPERTS_PER_GPU", 2)
def get_coplac_latency_json() -> str | None:
    return _get_str("RE_COPLAC_LATENCY_JSON", "") or None
```

`config.py`：
```python
@dataclass(frozen=True)
class CoPlacEConfig:
    devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    gpu0_ratio: float = 0.15
    gpu1_ratio: float = 0.15
    dram_ratio: float = 0.40
    ssd_path: str | None = None
    ptr_target: float = 0.99
    min_experts_per_gpu: int = 2
    min_depth: int = 1
    max_depth: int = 4
    @property
    def ssd_ratio(self) -> float:
        return max(0.0, 1.0 - (self.gpu0_ratio + self.gpu1_ratio + self.dram_ratio))
    @property
    def num_tiers(self) -> int:
        return 4 if self.ssd_path else (3 if len(self.devices) > 1 else 2)
    def tier_counts(self, num_experts: int) -> list[int]:
        """按比例切出每层专家数；余数补给 DRAM；总数守恒（单测断言）。
        GPU 下限修正：某 GPU tier 数 < min_experts_per_gpu 时从 DRAM 借位补齐（Mixtral 8 专家/层：floor 后 [1,1,4,2] → GPU0/GPU1 各从 DRAM 借 1 → [2,2,2,2]；OLMoE 64/DeepSeek 256 专家等模型 GPU 数天然 ≥ 下限，不受影响）。"""
```
`__init__.py` 导出 `CoPlacEConfig` 及后续模块。

**验证:** `pytest -q tests/test_coplac_config.py`（新建）：`tier_counts` 守恒、`ssd_ratio` 边界（ratio 和 >1 → ssd=0）、`num_tiers` 三档、GPU 下限修正（Mixtral 8 专家 → [2,2,4,0]；64 专家 → 下限不生效，纯比例）。

### Task 2: 共激活矩阵 + 社区布局（1h）

**Files:** Create `reframework/coplace/coactivation.py`；Test `tests/test_coplac_coactivation.py`

```python
def load_activation_log(path: str) -> dict[int, list[frozenset[int]]]:
    """jsonl {"step","layer","experts":[..]} -> {layer: [frozenset(experts) per step]}"""

def coactivation_matrix(per_layer_steps, num_experts) -> np.ndarray:
    """同 step 共现计数（不含对角），每层一个 [E,E]"""

def community_order(coact: np.ndarray) -> list[list[int]]:
    """优先 networkx louvain；ImportError 时退化为按行和降序的贪心分块（块大小=E//4 起步）"""
```
**数据:** 用 `litemoe/data/activation_logs/tiny.jsonl`（已存在）做真实数据单测；另造合成数据验证 `community_order` 在"两个稠密社区"输入下把社区内元素排进相邻块。
**验证:** 断言 tiny 日志的矩阵对称、对角为 0；社区输出是全体 expert 的一个划分（partition）。

### Task 3: SSDLatencyModel + MockSSDBackend（1h）

**Files:** Create `reframework/coplace/latency.py`、`reframework/coplace/ssd_tier.py`（先 Mock 版）；Test `tests/test_coplac_latency.py`

```python
class SSDLatencyModel:
    fast_mu = math.log(101e-6); fast_sigma = 0.3; fast_weight = 0.995
    slow_mean = 3e-3; slow_std = 2e-3; slow_weight = 0.005
    def cdf(self, t) -> float: ...
    def sample(self, rng=None) -> float: ...   # 混合采样
    def quantile(self, p) -> float: ...        # 二分求逆 CDF（[1e-6, 0.1] 区间）
    @classmethod
    def from_json(cls, path) -> "SSDLatencyModel": ...  # 覆盖默认参数（目标机 fio 回填）
```
**验证:** `quantile(0.999)` 落在 [2.0ms, 3.5ms]（对默认参数）；`cdf` 单调；10k 采样 P50≈101µs（±15%）。`MockSSDBackend`：`read()` 返回权重副本 + `model.sample()` 延迟。

### Task 4: PlacementPlanner（1.5h）

**Files:** Create `reframework/coplace/placement.py`；Test `tests/test_coplac_placement.py`

```python
class PlacementPlanner:
    def plan(self, num_experts: int, freq: np.ndarray | None = None,
             coact: np.ndarray | None = None, cfg: CoPlacEConfig | None = None) -> dict[int, int]:
        """strategy: freq（按激活频率降序切段，文档基线）/ coactivation（社区整块进同一 tier，
        社区按平均频率排段）/ joint（=coactivation 布局 + 频率决定段序；Task 9 联合优化前的等价物）。
        返回 {expert_id: Tier}，tier 顺序 [GPU0, GPU1, DRAM, SSD]（单卡时跳过 GPU1）。
        不变式：各段大小 == cfg.tier_counts；expert 不重不漏。"""
```
**验证:** 三种策略都满足守恒不变式；coactivation 策略下"高共激活对"尽量同 tier（断言同社区同 tier 对数 ≥ 随机基线）；`freq=None` 时退化为 id 排序（确定性）。

### Task 5: DeviceShuttle（Cpu mock + Cuda）（1.5h）

**Files:** Create `reframework/coplace/shuttle.py`；Test `tests/test_coplac_shuttle.py`
**语义依据：Task 0 真 GPU 冒烟报告** — non_blocking 重叠度、inflight 记账、扣减公式均以实测契约为准；若报告揭示差异，先修 Task 0 契约再实现。

CpuShuttle：`send` 返回 clone 并记 `inflight_bytes+=n`，`sync()` 清零；CudaShuttle 同接口但用 stream+event（**不在本 WSL 机跑 CUDA 断言**，构造时 `torch.cuda.is_available()` 为假直接 raise，单测只测 CpuShuttle 与 CudaShuttle 的字节记账逻辑——把记账抽成 `_Accounting` 基类在 CPU 上测）。
**验证:** 记账守恒（send n1+n2, sync 后 0）；CudaShuttle 在无 GPU 机上 `build_shuttle("cuda:0")` raise `RuntimeError`（不静默降级——文档 §2.2 教训：silent fallback 会掩盖配错的多卡运行，沿用 `EngineConfig._cuda` 的 raise 惯例）。

### Task 6: ExpertOffloadCache 扩展 — GPU1 池 + 带宽扣减 + RSE 钩子（2h，核心）

**Files:** Modify `reframework/moe/offload_cache.py`；Test `tests/test_coplac_cache.py`

1. dataclass 追加字段（全部默认 None/0，**现有 258 行测试必须全绿不动**）：
   `shuttle=None, lru1: OrderedDict|None, rse=None, inflight_ssd=0.0, inflight_pcie=0.0`
2. `reserve()`（:553）budget 行前：`budget -= (self.inflight_ssd + self.inflight_pcie)`，下限 0。
3. `_evict()` stale 候选排序处（:657/669 两个 `victim_of` 闭包）：若 `self.rse is not None` 用 `self.rse.rank(cands)` 替代 LRU 顺序，否则原路径。
4. GPU1 池：`lru1` 与 `_lru` 同构（OrderedDict[int, packed]）；`prefetch(ids, tier=0)` 增加 tier 参数选择目标池；`resident_set` 属性（:342）改为两池并集（保持 frozenset 返回类型）。
**验证（新测试）:**
- 带宽扣减：`inflight_ssd=0.5*bw*budget_ms` 时 `reserve` 返回 k 减半；
- RSE 钩子：注入 fake rse（返回固定顺序）验证 `_evict` 按其顺序驱逐；
- GPU1 池：`prefetch([x], tier=1)` 后 `x in resident_set` 且不在 `_lru`；
- **回归:** `pytest -q tests/test_moe_prefetch.py tests/test_moe_eviction_policy.py tests/test_litemoe_adapter.py` 全绿。

### Task 7: RSE 实现（1.5h）

**Files:** Create `reframework/coplace/rse.py`；Test `tests/test_coplac_rse.py`

```python
class RoutingAwareEviction:
    def __init__(self, sim_table: torch.Tensor | None, coact_row: np.ndarray | None,
                 next_predict_fn, lam_sim: float = 1.0): ...
    def rank(self, cands: list[int]) -> list[int]:
        """score = coact(e) + lam_sim * max_{r resident} sim[e][r]；升序返回（低分先驱逐）"""
```
**验证:** 高分专家（共激活大）排在后面；sim_table=None 时退化为纯共激活；next_predict_fn 给出的集合恒排最后（与 Task 6 的软保护一致）。

### Task 8: PTRPrefetchController + AdaptivePrefetchDepth（2h）

**Files:** Create `reframework/coplace/prefetch.py`；Test `tests/test_coplac_prefetch.py`

按 §4.6 契约实现。`trigger(layer_idx, next_set, is_prefill=False)` 内部：
```
if is_prefill: self.stats.prefill_skips += 1; return
missing = [e for e in next_set if not dram_hit(e) and tier(e)==TIER_SSD]
if not missing: return
lat = ssd.read(...)            # mock：同步
cache.inflight_ssd -= bytes    # 完成即归还带宽
self.depth.update(layer_times, [lat])
```
**验证:**
- prefill 批不触发（计数断言）；
- D 自适应：喂"延迟恒 0.9ms"流 → D 升到 max；喂"恒 0.1ms"流 → D 降到 min；
- 带宽扣减联动：trigger 后 `cache.inflight_ssd` 增减正确（mock 同步路径下为瞬时归还，用回调注入异步模拟）。

### Task 9: FusedMoE 双 GPU 分流 forward（2h）

**Files:** Modify `reframework/moe/moe_layer.py`；Test `tests/test_coplac_dual_gpu.py`

`FusedMoE.enable_dual_gpu(shuttle, placement, cache1)`：`forward` 在 `route()`（:226）得到 `topk_ids` 后：
```
gpu1_ids = [e for e in unique(topk_ids) if placement[e] == TIER_GPU1]
if gpu1_ids and self.shuttle:
    x1 = self.shuttle.send(x_subset, "cuda:1")      # token 级，非 All-to-All
    out1 = self.shuttle.recv(...)                    # mock 下 = 对端 cache 上算
    # GPU0 侧合并：out = out0 + router_w1 * out1
else: 原路径
```
mock 验证：构造 `placement={2: TIER_GPU1, 其余 TIER_GPU0}`，CpuShuttle，断言 (a) 分流路径输出与不分流逐元素相等（mock 下计算同源，误差 <1e-3 fp16）；(b) `shuttle.inflight_bytes` 在 forward 中 >0；(c) placement 全 GPU0 时零开销走原路径。
**注意:** `FusedMoE.forward` 现有三个分支（streaming/split/default，:357/:260/:429）——分流逻辑加在 **default 分支入口**，不动 streaming/split。

### Task 10: 引擎装配 `_setup_coplac()`（1.5h）

**Files:** Modify `reframework/engine/engine.py`（`_setup_moe_offload` 末尾调用）、`reframework/coplace/__init__.py` 加 `assemble(engine)`；Test `tests/test_coplac_engine.py`

```python
def assemble(engine) -> None:
    cfg = CoPlacEConfig.from_env()
    if not cfg:  # RE_COPLAC=0 或未配置 → no-op
    ssd = FileSSDBackend(cfg.ssd_path) if cfg.ssd_path else MockSSDBackend()
    shuttle = build_shuttle(cfg.devices)      # 无 GPU 机自动 CpuShuttle + 日志
    per-layer: planner.plan(...) -> placement.json（写 ckpt 目录）
    cache.rse = RoutingAwareEviction(sim_table, coact_row, pred.predict_next)
    cache.prefetch_ctrl = PTRPrefetchController(...)
    mlp.enable_dual_gpu(shuttle, placement, cache1)  if len(cfg.devices)>1
    engine.offload_report() 增加 CoPlacE 段（tier 分布/预取统计）
```
**验证:** `RE_COPLAC=1 python -c "build_engine(...tiny ckpt...)"` 在 CPU 机可构造（tiny MoE ckpt 复用 litemoe 测试数据或现有 tests 夹具）；`offload_report()` 输出含 "CoPlacE: tiers=..."；RE_COPLAC 未设时行为与现状完全一致（回归 `make test`）。

### Task 11: CrossDeviceRouterLoss + 路由微调脚本（2h）

**Files:** Create `reframework/coplace/router_loss.py`、`scripts/coplace_router_finetune.py`；Test `tests/test_coplac_router_loss.py`

loss 实现按方案 §4.5.2（Gumbel-Softmax 设备 one-hot 加权 → 设备熵均值 × λ），**修正文档代码两处**：(a) `F.one_hot` 的 num_classes 应为 tier 数（4）而非 expert 数；(b) softmax 应放在 gather **之后**、只在 selected_experts 维度上做——先 gather 出选中专家的 logits，再在 top-k 子集内 softmax 归一；文档"全专家 softmax 后 gather"给出的是全分布概率，不是选中子集上的归一化分布。此修正影响 Gumbel-Softmax 松弛的数学正确性（设备熵须基于重归一化分布计算），不是风格问题。
脚本：`--ckpt --placement coplace_placement.json --calib litemoe/data/activation_logs/*.jsonl`；交替优化 2-3 轮（plan → finetune gate → 重新统计 freq → replan），每轮输出跨设备传输率（token 选中 expert 的 tier 对数 / token 数），**目标：相对原始 router 降 30%+**（文档 §6.4 预期，作为验收断言写入测试）。
**验证:** loss 在"所有 token 只选同 tier 专家"时 = 0；跨 tier 激活时 >0 且随 λ 单调；soft_device 对每个 token 在选中子集上求和 = 1（断言子集归一化，即 (b) 修正）；微调脚本在 tiny 数据上跑通 2 轮（CPU，<2min）。

### Task 12: 端到端 + 目标机冒烟（1.5h）

**Files:** Create `scripts/coplace_hardware_check.py`、`tests/test_coplac_e2e.py`；Modify `Makefile`（加 `test-coplac` target）

- e2e（CPU）：tiny MoE 模型 + RE_COPLAC=1 + MockSSD，跑 20 step decode，断言 `OffloadStats` 新字段（`ssd_reads`、`ptr`、`cross_device_tokens`）有值且 ptr∈[0,1]。
- 冒烟（仅目标机）：检查双卡可见/驱动 ≥r580/CUDA≥13/`nvidia-smi topo -m` 确认无 NVLink；GPU0↔GPU1 PCIe 双向带宽实测（串行 vs 并行吞吐比 → 验证竞争假设，即 Task 0 双卡项在目标机执行）。
- **SSD 稳态前/后对比分支（必须）**：目标 SSD 近乎全新（percentage_used=0%，power_on_hours=5），直接 fio 得到的是"最佳状态"数据，审稿人会质疑可重复性。脚本内置流程：
  1. `nvme smart-log`：percentage_used < 5% 或 power_on_hours < 24h → 标记未稳态，**拒绝回填正式参数**（exit non-zero，打印 full-disk 写入指引）；
  2. `--stabilize`：先顺序写 500GB（目标 SSD 上临时文件）使盘进入稳态，再重测读套件；
  3. 输出**稳态前/后对比**（P50/P99/P99.9）并写入 `coplace_latency.json` 的 `steady_state: {before, after, used: "after"}`；正式回填只用稳态后数据，报告必须含 before/after 差值。
- Makefile:
```
test-coplac:
	$(PYTHON) -m pytest -q tests/test_coplac_*.py
```

## 6. 验证总闸（Definition of Done）

1. `make test`（= `pytest -q` 全量）绿：现有 3 个测试文件 + 9 个新 test_coplac_* 文件。
2. `RE_COPLAC` 未设时 diff 行为为零（回归保护）。
3. 目标机 `scripts/coplace_hardware_check.py` 输出存档到 `docs/coplace_hardware_$(date).txt`。
4. 每个 commit 只含对应 Task 的文件；不触碰既有 staged/modified 文件。

## 7. 风险与开放问题

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 本 WSL 机无第二张 GPU，双卡逻辑只能 mock 验证 | CpuShuttle 全覆盖逻辑；CudaShuttle 仅目标机冒烟；文档中明确标注验证边界 |
| R2 | FreeToken 本体不在仓库，"FTW 格式转换"无法在本仓库落地 | 本计划用 safetensors+int8（现有 `quantize_int8_host`）作为权重表示；FTW 转换留给目标机 FreeToken 工具链，`scripts/coplace_*` 保持格式无关 |
| R3 | 共激活图社区划分无 networkx | 贪心回退已设计（Task 2）；若社区质量差，RSE 退化回 LRU 排序（钩子可关） |
| R4 | PCIe 双向竞争假设（send/recv 共享链路）可能不成立（P2P 方向分离） | 冒烟脚本实测后回填 `_Accounting` 策略（串行/并行记账开关） |
| R5 | 激活日志（tiny/huihui）规模小，放置质量存疑 | 目标机用真实 Mixtral 跑 calibration 生成日志后再 plan；测试只验不变式不验最优性 |
| R6 | 文档 §4.5.2 代码有 2 处数学错误（已修正，见 Task 11） | 单测断言覆盖两个修正点 |
| R7 | SSD 稳态化需数小时 full-disk 写入 | 冒烟脚本只**检测并提示**，不自动执行（破坏性操作必须用户确认） |
| R8 | `ExpertOffloadCache` 是 dataclass，加字段需保持 `from_config` 兼容 | 新字段全默认值；`from_config`（:269）不改签名，CoPlacE 装配走 `assemble(engine)` 后置注入 |
| R9 | mock 异步语义（non_blocking 重叠/inflight 记账）可能与真实 CUDA 路径不符 | Task 0 前置冒烟实测固化契约；真路径不符时先修契约再写 mock |

## 8. 与三阶段开源节奏的对应

- **第一阶段（核心框架）:** Task 0（前置闸门）, 1-6, 10 → 双 GPU 缓存分裂 + SSD 第四层
- **第二阶段（RSE + PTR）:** Task 7, 8 → 路由感知替换 + 预取深度自适应
- **第三阶段（微调 + 实验）:** Task 11, 12 → 路由微调工具链 + 实验脚本（`coplace_hardware_check.py`、`coplace_router_finetune.py`）

许可证：**保持 MIT（2026-09-26 用户确认）**——reframework 是本仓库原创代码（非 FreeToken 代码拷贝），CoPlacE 扩展同样原创；方案文档"继承 FreeToken Apache-2.0"的措辞据此修订。不改 LICENSE 文件、不改 pyproject license 字段；第一阶段发布时 README 加一行 license 说明（reframework/CoPlacE = MIT，FreeToken 上游 = Apache-2.0，两者独立）。
