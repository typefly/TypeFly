import os
from enum import Enum
from openai import OpenAI

class ModelType(Enum):
    GPT4 = "gpt-4"
    GPT4O = "gpt-4o"
    GPT5 = "gpt-5"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_FILE = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

# Network timeout (seconds) for the OpenAI client. Prevents a stalled request
# from hanging the plan worker indefinitely. (Retry/backoff is added in P1.)
REQUEST_TIMEOUT = 30.0

class LLMWrapper:
    """
    A wrapper for the LLM API.
    """
    def __init__(self, temperature: float=0.1):
        self.temperature = temperature
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Do not echo the key or imply it might be printed elsewhere.
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.gpt_client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)

    def request(self, prompt, model_type: ModelType | str, json_mode: bool = False) -> str:
        """
        Request the LLM API with the prompt and model type.

        ``json_mode`` forces the response to be a single JSON object (used for
        planning). It must stay False for free-form replies such as ``probe``,
        which returns plain text like ``bottle`` / ``True`` / ``2``.
        """

        if model_type in [ModelType.GPT4, ModelType.GPT4O, ModelType.GPT5]:
            kwargs = dict(
                model=model_type.value if isinstance(model_type, ModelType) else model_type,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=self.temperature,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.gpt_client.chat.completions.create(**kwargs)
            ret = response.choices[0].message.content
        # elif other models, implement here
        else:
            raise ValueError(f"Model type {model_type} not supported.")

        self._maybe_log(prompt, ret)
        return ret

    @staticmethod
    def _maybe_log(prompt: str, response: str) -> None:
        """Append the prompt/response to chat_log.txt only when explicitly
        enabled (TYPEFLY_CHAT_LOG), and create the file 0600 so the captured
        prompts/scene descriptions are not world-readable.

        NOTE: this log is unbounded and unredacted; structured, rotating,
        redacted logging replaces it in P2.
        """
        if not os.environ.get("TYPEFLY_CHAT_LOG"):
            return
        try:
            fd = os.open(CHAT_LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(prompt + "\n---\n")
                f.write((response or "") + "\n--------------------------------\n")
        except OSError:
            pass
