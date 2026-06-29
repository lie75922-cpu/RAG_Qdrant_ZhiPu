"""Zhipu chat completion adapter."""

from __future__ import annotations

from typing import Any

from ragforgex.generators.openai_compatible_generator import OpenAICompatibleGenerator


class ZhipuGenerator(OpenAICompatibleGenerator):
    def __init__(
        self,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        model: str = "glm-4-flash",
        **kwargs: Any,
    ) -> None:
        super().__init__(base_url=base_url, model=model, **kwargs)

