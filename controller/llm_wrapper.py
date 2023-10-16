import os, json, ast
import openai

openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')
# MODEL_NAME = "gpt-3.5-turbo-16k"
MODEL_NAME = "gpt-4"

class LLMWrapper:
    def __init__(self, temperature=0.05):
        self.temperature = temperature

    def query(self, prompt, model_name=MODEL_NAME):
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # save the message in a txt
        with open("./assets/chat_log.txt", "a") as f:
            f.write(prompt + "\n##################################\n")
            f.write(json.dumps(response) + "\n##################################\n")

        return response["choices"][0]["message"]["content"]