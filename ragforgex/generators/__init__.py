from ragforgex.core.registry import GENERATORS
from ragforgex.generators.ollama_generator import OllamaGenerator
from ragforgex.generators.openai_compatible_generator import OpenAICompatibleGenerator
from ragforgex.generators.zhipu_generator import ZhipuGenerator

GENERATORS.register("openai_compatible", OpenAICompatibleGenerator)
GENERATORS.register("zhipu", ZhipuGenerator)
GENERATORS.register("ollama", OllamaGenerator)
GENERATORS.register("echo", OpenAICompatibleGenerator)

