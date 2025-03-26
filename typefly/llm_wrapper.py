import os
from enum import Enum
from openai import OpenAI, Stream, ChatCompletion

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_FILE = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    def __init__(self, temperature: float=0.0):
        self.temperature = temperature
        self.gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def request(self, prompt, model_type: ModelType, stream: bool=False) -> str | Stream[ChatCompletion.ChatCompletionChunk]:        
        response = self.gpt_client.chat.completions.create(
            model=model_type.value,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=stream,
        )

        with open(CHAT_LOG_FILE, "a") as f:
            f.write(prompt + "\n---\n")
            if not stream:
                f.write(response.model_dump_json(indent=2) + "\n---\n")

        if stream:
            return response
        return response.choices[0].message.content