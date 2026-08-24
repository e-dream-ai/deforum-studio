from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple


_LORA_TAG = re.compile(
    r"<lora:(?P<alias>[^:<>]+):(?P<strength>[+-]?(?:\d+(?:\.\d*)?|\.\d+))>",
    re.IGNORECASE,
)
_MAX_ABS_STRENGTH = 10.0


@dataclass(frozen=True)
class LoraRequest:
    alias: str
    strength: float


@dataclass(frozen=True)
class LoraSpec:
    alias: str
    path: Path
    strength_model: float
    strength_clip: float


class LoraRegistry:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)
        self._paths = self._discover()

    def resolve(self, request: LoraRequest) -> LoraSpec:
        path = self._paths.get(request.alias.casefold())
        if path is None:
            available = ", ".join(sorted(item.stem for item in self._paths.values())) or "none"
            raise ValueError(f"Unknown LoRA '{request.alias}'. Available LoRAs: {available}")
        return LoraSpec(
            alias=request.alias,
            path=path,
            strength_model=request.strength,
            strength_clip=request.strength,
        )

    def _discover(self) -> Dict[str, Path]:
        if not self._directory.is_dir():
            return {}

        discovered: Dict[str, Path] = {}
        for path in sorted(self._directory.glob("*.safetensors")):
            alias = path.stem.casefold()
            if alias in discovered:
                raise ValueError(f"Duplicate LoRA alias '{path.stem}' in {self._directory}")
            discovered[alias] = path.resolve()
        return discovered


def parse_lora_prompt(prompt: str) -> Tuple[str, Tuple[LoraRequest, ...]]:
    requests: Dict[str, LoraRequest] = {}

    def remove_tag(match: re.Match) -> str:
        alias = match.group("alias").strip()
        strength = float(match.group("strength"))
        if not alias:
            raise ValueError("LoRA alias cannot be empty")
        if not math.isfinite(strength) or abs(strength) > _MAX_ABS_STRENGTH:
            raise ValueError(
                f"LoRA strength for '{alias}' must be between -{_MAX_ABS_STRENGTH:g} and {_MAX_ABS_STRENGTH:g}"
            )

        key = alias.casefold()
        request = LoraRequest(alias=alias, strength=strength)
        existing = requests.get(key)
        if existing is not None and existing.strength != strength:
            raise ValueError(f"LoRA '{alias}' has conflicting strengths in one prompt")
        requests[key] = request
        return ""

    cleaned_prompt = _LORA_TAG.sub(remove_tag, prompt).strip()
    if "<lora:" in cleaned_prompt.casefold():
        raise ValueError("Invalid LoRA tag; expected <lora:alias:strength>")
    ordered = tuple(sorted(requests.values(), key=lambda item: item.alias.casefold()))
    return cleaned_prompt, ordered


def prepare_lora_prompts(
    prompts: Sequence[str],
    lora_directory: str,
) -> Tuple[Tuple[str, ...], Tuple[LoraSpec, ...]]:
    parsed = tuple(parse_lora_prompt(prompt) for prompt in prompts)
    if not parsed:
        return (), ()

    expected = parsed[0][1]
    if any(requests != expected for _, requests in parsed[1:]):
        raise ValueError("All Deforum keyframes must use the same LoRAs and strengths")

    registry = LoraRegistry(lora_directory)
    cleaned_prompts = tuple(prompt for prompt, _ in parsed)
    specs = tuple(registry.resolve(request) for request in expected)
    return cleaned_prompts, specs
