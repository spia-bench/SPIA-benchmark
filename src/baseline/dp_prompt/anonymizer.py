"""
DP-Prompt: Locally Differentially Private Document Generation
Zero-shot paraphrasing for text anonymization
"""

from typing import Optional
import warnings
from src.llm.client import call_api_with_param, set_api_provider


class DPPromptAnonymizer:
    """
    DP-Prompt anonymizer using zero-shot paraphrasing.
    """

    def __init__(
        self,
        api_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 1.5
    ):
        """
        Initialize DP-Prompt anonymizer.

        Args:
            api_provider: API provider ("openai", "anthropic", "ollama", or "local")
            model_name: Model name (e.g., "gpt-3.5-turbo", "google/flan-t5-xl", etc.)
            temperature: Sampling temperature (1.0 recommended, higher = more privacy)
        """
        self.api_provider = api_provider
        self.model_name = model_name
        self.temperature = temperature

        self.model = None
        self.tokenizer = None

        if api_provider == "local":
            self._load_local_model(model_name)
        else:
            from src.llm.client import set_api_provider
            set_api_provider(api_provider, model_name)

    def _load_local_model(self, model_name: str):
        """Load local Hugging Face model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Flan-T5 is Seq2Seq model, StableLM is CausalLM
        if "t5" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        print("Model loaded successfully!")

    @staticmethod
    def create_prompt(text: str) -> str:
        return f"Document: {text}\nParaphrase of the document:"

    def anonymize(
        self,
        text: str,
        temperature: Optional[float] = None
    ) -> str:
        """
        Anonymize a single text using DP-Prompt.
        """
        temp = temperature if temperature is not None else self.temperature

        # Anthropic API only supports temperature 0~1
        if self.api_provider == "anthropic" and temp > 1.0:
            temp = 1.0

        prompt = self.create_prompt(text)

        # Calculate dynamic max_tokens based on input text length
        # Estimate: ~2 chars per token (rough approximation)
        estimated_tokens = len(text) // 2
        max_tokens = int(estimated_tokens * 1.5) + 50  # 1.5x input + buffer

        if self.api_provider == "local":
            return self._anonymize_local(prompt, temp, max_tokens)
        else:
            from src.llm.client import call_api_with_param
            try:
                response = call_api_with_param(
                    prompt,
                    temperature=temp,
                    top_p=1.0,
                    top_k=0,
                    max_tokens=max_tokens
                )
                return response.strip()
            except Exception as e:
                warnings.warn(f"LLM API error: {e}")
                return ""

    def _anonymize_local(self, prompt: str, temperature: float, max_new_tokens: int = 150) -> str:
        """Anonymize using local Hugging Face model."""
        import torch

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Set pad_token_id to suppress warning for models without pad token
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id

            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id
            )

            # Seq2Seq model returns full output, CausalLM excludes prompt
            if "t5" in self.model_name.lower():
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                generated = outputs[:, inputs["input_ids"].shape[1]:]
                result = self.tokenizer.decode(generated[0], skip_special_tokens=True)

            return result.strip()

        except Exception as e:
            warnings.warn(f"Local model error: {e}")
            return ""
