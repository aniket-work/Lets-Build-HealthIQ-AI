# llm.py
from typing import Optional
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LLMManager:
    """
    Manager class for Ollama-based LLM operations.
    This class handles the initialization and management of the Ollama language model,
    providing a clean interface for model interactions.
    """

    def __init__(
            self,
            model_name: str = "llama2:3b",
            temperature: float = 0.3,
            max_tokens: int = 2048,
            top_p: float = 1.0,
            base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the LLM manager with Ollama-specific parameters.

        Args:
            model_name: Name of the Ollama model to use
            temperature: Controls randomness in the output
            max_tokens: Maximum number of tokens to generate
            top_p: Cumulative probability for top-p sampling
            base_url: URL of the Ollama API endpoint
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = base_url
        self._llm: Optional[Ollama] = None

    @property
    def llm(self) -> Ollama:
        """
        Lazy load the Ollama model.
        Returns:
            Ollama: Initialized Ollama model instance
        """
        if self._llm is None:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            self._llm = Ollama(
                model=self.model_name,
                temperature=self.temperature,
                num_ctx=self.max_tokens,
                top_p=self.top_p,
                base_url=self.base_url,
                callbacks=callback_manager
            )
        return self._llm

    def reset_model(self):
        """
        Reset the model instance.
        Useful when you need to change parameters or clear the model's state.
        """
        self._llm = None