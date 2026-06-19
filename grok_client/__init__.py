from importlib.metadata import version

from .client import GrokClient
from .grok_openai_client import GrokOpenAIClient

__version__ = version("GrokWeb-to-API")
__all__ = ["GrokClient", "GrokOpenAIClient"]
