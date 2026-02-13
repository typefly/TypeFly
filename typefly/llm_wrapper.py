import os
from enum import Enum
from openai import OpenAI

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"
    GPT5 = "gpt-5"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_FILE = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    """
    A wrapper for the LLM API.
    """
    def __init__(self, temperature: float=0.1):
        self.temperature = temperature
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in the environment variable or in the .env file.")
        
        self.gpt_client = OpenAI(api_key=api_key)

    def request(self, prompt, model_type: ModelType | str) -> str:        
        """
        Request the LLM API with the prompt and model type.
        """

        if model_type in [ModelType.GPT4, ModelType.GPT4O, ModelType.GPT5]:
            response = self.gpt_client.chat.completions.create(
                model=model_type.value if isinstance(model_type, ModelType) else model_type,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            ret = response.choices[0].message.content
        # elif other models, implement here
        else:
            raise ValueError(f"Model type {model_type} not supported.")

        # log the prompt and response
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(prompt + "\n---\n")
            f.write(ret + "\n--------------------------------\n")

        return ret