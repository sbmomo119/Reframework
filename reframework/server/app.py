"""Minimal FastAPI server exposing an OpenAI-compatible chat endpoint.

Single-stream, fp32, no CUDA graph. Each HTTP request maps onto one engine
generation. The engine is not concurrently batched in this phase (the
``scheduler/`` package will change that); the server serializes requests
through the single engine instance, which is correct for a single-stream
engine. Both one-shot and SSE streaming responses are supported so the same
client code used against an OpenAI API works unchanged.

Run::

    python -m reframework.server.app /path/to/ckpt --host 127.0.0.1 --port 8000

Endpoints:
  * ``GET  /healthz``
  * ``GET  /v1/models``
  * ``POST /v1/chat/completions``   (``stream`` optional)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from queue import Queue
from typing import List, Optional

from reframework.core import SamplingParams
from reframework.engine import build_engine, EngineConfig
from reframework.tokenizer import Tokenizer
from reframework.utils.logging import init_logger

logger = init_logger(__name__)

__all__ = ["create_app", "build_app", "main"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _map_sampling(body: dict) -> SamplingParams:
    """Translate OpenAI-style sampling fields into a ``SamplingParams``."""
    temperature = float(body.get("temperature", 0.0))
    top_p = float(body.get("top_p", 1.0))
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 128)
    seed = body.get("seed")
    top_k = int(body.get("top_k", 1))
    if temperature <= 1e-6:
        # greedy: force the argmax path
        top_k, top_p = 1, 1.0
    return SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_tokens,
        seed=None if seed is None else int(seed),
    )


def _encode_prompt(tok: Tokenizer, messages: List[dict]) -> List[int]:
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    bos = getattr(tok, "bos_token_id", None)
    while ids and bos is not None and ids[0] == bos:
        ids = ids[1:]
    return ids


def _usage(prompt_n: int, completion_n: int) -> dict:
    return {
        "prompt_tokens": prompt_n,
        "completion_tokens": completion_n,
        "total_tokens": prompt_n + completion_n,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(
    model_path: str,
    *,
    device: str = "auto",
    max_seq_len: int = 4096,
    model_name: Optional[str] = None,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="Re", version="0.1.0")

    tok = Tokenizer(model_path)
    engine = build_engine(
        model_path,
        tokenizer=tok,
        cfg=EngineConfig(device=device, max_seq_len=max_seq_len),
    )
    name = model_name or tok.name_or_path

    # ------------------------------------------------------------------ meta
    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "device": str(engine.device)}

    @app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "re",
                }
            ],
        }

    # -------------------------------------------------------- chat completions
    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="`messages` is required")
        sp = _map_sampling(body)
        stream = bool(body.get("stream", False))
        prompt_ids = _encode_prompt(tok, messages)

        rid = str(uuid.uuid4())
        created = int(time.time())

        if not stream:
            loop = asyncio.get_running_loop()
            t0 = time.time()
            out_ids = await loop.run_in_executor(None, engine.generate, prompt_ids, sp)
            text = tok.decode(out_ids, skip_special_tokens=True)
            logger.info("chat: rid=%s in=%d out=%d %.2fs", rid, len(prompt_ids), len(out_ids), time.time() - t0)
            return JSONResponse(
                {
                    "id": rid,
                    "object": "chat.completion",
                    "created": created,
                    "model": name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage(len(prompt_ids), len(out_ids)),
                }
            )

        return StreamingResponse(_sse(engine, tok, name, rid, created, prompt_ids, sp), media_type="text/event-stream")

    return app


# ---------------------------------------------------------------------------
# SSE streaming: a producer thread drives the blocking engine token-by-token
# onto a queue; the async generator drains it and yields OpenAI-style chunks.
# ``delta.content`` carries the *incremental* text (re-decoded with a prefix
# diff so subword merges stay coherent).
# ---------------------------------------------------------------------------
async def _sse(engine, tok, model: str, rid: str, created: int, prompt_ids: List[int], sp: SamplingParams):
    q: Queue = Queue()

    def _produce() -> None:
        try:
            engine.generate(prompt_ids, sp, on_token=lambda t: q.put(t))
        finally:
            q.put(None)  # sentinel

    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, _produce)

    out_ids: List[int] = []
    decoded_so_far = ""
    while True:
        t = await loop.run_in_executor(None, q.get)
        if t is None:
            break
        out_ids.append(int(t))
        full = tok.decode(out_ids, skip_special_tokens=True)
        delta = full[len(decoded_so_far):] if full.startswith(decoded_so_far) else full
        decoded_so_far = full
        chunk = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    final = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
        ],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
    await fut


def build_app(
    model_path: str,
    *,
    device: str = "auto",
    max_seq_len: int = 4096,
    model_name: Optional[str] = None,
):
    return create_app(model_path, device=device, max_seq_len=max_seq_len, model_name=model_name)


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Re OpenAI-compatible server")
    p.add_argument("model", help="path to a HF checkpoint dir")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--model-name", default=None)
    args = p.parse_args(argv)

    import uvicorn

    app = build_app(args.model, device=args.device, max_seq_len=args.max_seq_len, model_name=args.model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
