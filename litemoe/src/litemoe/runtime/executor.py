"""Execution loop for ``litemoe run``: wires store + cache + predictor + model
into a timed prefill/decode run, logs activations, and returns a
:class:`RunReport`."""

from __future__ import annotations

import json
import time
from typing import List, Optional

import torch

from litemoe.config import LitemoeConfig, parse_dtype, parse_device
from litemoe.interface import make_cache
from litemoe.model.model import Qwen3MoEModel, _sample
from litemoe.predictor.ngram import NgramPredictor
from litemoe.runtime.profiler import Profiler, RunReport
from litemoe.store import make_store


class Executor:
    """Builds the runtime from a :class:`LitemoeConfig` and runs generation."""

    def __init__(self, cfg: LitemoeConfig):
        self.cfg = cfg

    def _load(self):
        import glob
        import os
        dtype = parse_dtype(self.cfg.compute.dtype)
        device = parse_device(self.cfg.compute.device)
        path = self.cfg.model.path
        # 目录 -> 自动找第一个 *.safetensors（与 make_store 保持一致），
        # 这样相邻的 config.json 也能被找到
        if os.path.isdir(path):
            cands = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            if not cands:
                raise FileNotFoundError(f"no *.safetensors in {path!r}")
            path = cands[0]
        # safetensors models need the sibling config.json for the meta
        config = None
        if path.lower().endswith((".safetensors", ".st")):
            cfgj = os.path.join(os.path.dirname(path) or ".", "config.json")
            if os.path.exists(cfgj):
                with open(cfgj, "r", encoding="utf-8") as fh:
                    config = json.load(fh)
        t0 = time.perf_counter()
        store = make_store(path, config=config)
        cache = make_cache(store, strategy=self.cfg.cache.strategy,
                           max_experts=self.cfg.cache.max_experts,
                           predict_window=self.cfg.cache.predict_window,
                           device=str(device))
        rope_theta = float(store.meta.extra.get("rope_theta", 10000.0))
        rms_eps = float(store.meta.extra.get("rms_norm_eps", 1e-6))
        model = Qwen3MoEModel(store, cache=cache, dtype=dtype, device=device,
                              rope_theta=rope_theta, rms_eps=rms_eps)
        self.load_s = time.perf_counter() - t0
        return store, cache, model

    def run(self, prompt: Optional[str] = None, max_new_tokens: Optional[int] = None,
            do_sample: bool = False, temperature: float = 1.0, top_p: float = 0.9) -> RunReport:
        cfg = self.cfg
        prompt = prompt if prompt is not None else cfg.prompt
        max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens

        store, cache, model = self._load()
        meta = store.meta
        device = model.device
        n_layers = int(meta.n_layers)
        top_k = int(meta.top_k)

        # tokenizer (optional): encode/decode the prompt
        tokenizer = self._tokenizer()
        if prompt and prompt.strip():
            input_ids = self._encode(tokenizer, prompt, model)
        else:
            # no prompt -> use a short deterministic seed sequence
            input_ids = torch.tensor([[1, 405, 2990, 345]], device=device, dtype=torch.long)

        predict_window = cfg.cache.predict_window
        # The n-gram predictor is always instantiated (cheap); it feeds the
        # PDE cache later and its observation history is useful for analysis.
        predictor = NgramPredictor(n_layers, window=predict_window)

        prof = Profiler(cache=cache, predict_window=predict_window,
                        log_activations=cfg.metrics.log_activations,
                        activations_file=cfg.metrics.activations_file)
        prof.open_log()
        prof.begin()
        torch.manual_seed(cfg.seed)

        # ---- prefill -----------------------------------------------------
        prompt_ids = input_ids.to(device)
        positions = torch.arange(prompt_ids.size(1), device=device)
        t0 = time.perf_counter()
        logits, routed = model.forward(prompt_ids, positions)
        ttft = prof.prefill_done(t0)
        for L in range(n_layers):
            prof.log_activation(0, L, [int(x) for x in routed[L][0].tolist()])

        # feed the predictor with the prompt-token history (best-effort)
        if predictor is not None:
            for i, tok in enumerate([int(x) for x in prompt_ids[0].tolist()]):
                predictor.note(i, tok)

        # ---- decode ------------------------------------------------------
        new_ids = [int(prompt_ids[0, -1].item())]
        seq = prompt_ids
        next_logits = logits[:, -1, :]
        prof.mark_decode_start()
        for step in range(1, max_new_tokens + 1):
            if do_sample:
                nxt = _sample(next_logits, temperature, top_p)
            else:
                nxt = next_logits.argmax(-1, keepdim=True)
            new_tok = int(nxt.item())
            new_ids.append(new_tok)
            if do_sample and new_tok == int(getattr(model, "eos_id", -1)):
                break
            seq = torch.cat([seq, nxt.to(device).to(torch.long)], dim=1)
            pos = torch.tensor([seq.size(1) - 1], device=device)
            logits, routed = model.forward(nxt.to(device).to(torch.long), pos)
            step_ms = prof.decode_step_ms()
            for L in range(n_layers):
                eids = [int(x) for x in routed[L][0].tolist()]
                prof.log_activation(step, L, eids)
                if predictor is not None:
                    predictor.observe(step, L, eids)
            if predictor is not None:
                # observe the freshly sampled token for the next step's context
                predictor.note(step, new_tok)
            next_logits = logits[:, -1, :]
            if cfg.metrics.print_per_step:
                s = cache.stats()
                hr = s.hit_rate
                print(f"[step {step:3d}] {step_ms:8.2f} ms  "
                      f"hit-rate {hr:5.3f}  evict {s.evictions}  "
                      f"tok={new_tok}", flush=True)

        prof.flush_log()
        prof.close_log()
        output_ids = new_ids
        output_text = self._decode(tokenizer, output_ids)
        report = prof.report(prompt_tokens=input_ids.size(1), output_ids=output_ids,
                             output_text=output_text)
        report.__dict__["load_s"] = self.load_s  # attach (not in dataclass)
        return report

    # -- tokenizer helpers -----------------------------------------------
    def _tokenizer(self):
        import os
        path = self.cfg.model.tokenizer
        # 未配置则回退到 model.path 同目录
        if not path:
            path = os.path.dirname(self.cfg.model.path) or "."
        # 如果给的是目录，拼接 tokenizer.json / tokenizer.model
        if os.path.isdir(path):
            for cand in ("tokenizer.json", "tokenizer.model"):
                p = os.path.join(path, cand)
                if os.path.exists(p):
                    path = p
                    break
        if not path or not os.path.isfile(path):
            return None
        try:
            from tokenizers import Tokenizer
            return Tokenizer.from_file(path)
        except Exception as e:
            import warnings
            warnings.warn(f"tokenizer load failed: {e}")
            return None

    def _encode(self, tokenizer, prompt: str, model) -> torch.Tensor:
        if tokenizer is not None:
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            return torch.tensor([ids], device=model.device, dtype=torch.long)
        # fallback: naive space-separated ints (no tokenizer available)
        toks = []
        for w in prompt.split():
            toks.append(int(w) if w.lstrip("-").isdigit() else 0)
        if not toks:
            toks = [1, 405, 2990, 345]
        return torch.tensor([toks], device=model.device, dtype=torch.long)

    def _decode(self, tokenizer, ids: List[int]) -> str:
        if tokenizer is not None:
            try:
                return tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                pass
        return ""


def run_config(path: str, **overrides) -> RunReport:
    """Convenience: load a config file and run it with optional overrides."""
    cfg = LitemoeConfig.from_file(path)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    return Executor(cfg).run(**{k: v for k, v in overrides.items()
                                if k in ("prompt", "max_new_tokens", "do_sample",
                                         "temperature", "top_p") and v is not None})
