"""HF tokenizer wrapper (transformers AutoTokenizer).

Thin on purpose: the engine only needs ``encode`` / ``decode`` plus the
eos/pad ids, and the CLI wants a chat template. Everything else stays in
``transformers``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


class Tokenizer:
    """A text <-> ids wrapper over ``transformers.AutoTokenizer``."""

    def __init__(self, name_or_path: str, *, trust_remote_code: bool = False) -> None:
        from transformers import AutoTokenizer  # lazy: keep module import light

        self._tk = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        self.name_or_path = name_or_path

    # ------------------------------------------------------------ vocab / ids
    @property
    def vocab_size(self) -> int:
        return int(self._tk.vocab_size)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tk.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._tk.pad_token_id

    @property
    def has_chat_template(self) -> bool:
        return bool(self._tk.chat_template)

    # --------------------------------------------------------------- encode
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [int(t) for t in self._tk.encode(text, add_special_tokens=add_special_tokens)]

    # --------------------------------------------------------------- decode
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tk.decode([int(t) for t in token_ids], skip_special_tokens=skip_special_tokens)

    # -------------------------------------------------------- chat templates
    def apply_chat_template(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render ``[{role, content}, ...]`` to a prompt string.

        Uses the tokenizer's own ``chat_template`` when present (Llama 3,
        Qwen3 ship one); otherwise a minimal ``User:/Assistant:`` fallback.
        """
        if self.has_chat_template:
            return self._tk.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        lines = []
        for m in messages:
            role = "User" if m["role"] in ("user", "human") else "Assistant"
            lines.append(f"{role}: {m['content']}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    def __call__(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.encode(text, add_special_tokens=add_special_tokens)


__all__ = ["Tokenizer"]
