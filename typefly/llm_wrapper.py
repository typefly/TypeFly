import os
from enum import Enum
from openai import OpenAI, Stream, ChatCompletion

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"
    GPT5 = "gpt-5"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_FILE = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    def __init__(self, temperature: float=0.1):
        self.temperature = temperature
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in the environment variable or in the .env file.")
        self.gpt_client = OpenAI(api_key=api_key)

    def request(self, prompt, model_type: ModelType | str, stream: bool=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:        
        response = self.gpt_client.chat.completions.create(
            model=model_type.value if isinstance(model_type, ModelType) else model_type,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
        )

        with open(CHAT_LOG_FILE, "a") as f:
            f.write(prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response
        return response.choices[0].message.content