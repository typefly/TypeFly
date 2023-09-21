import os
import openai
openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "gpt-3.5-turbo-16k"

class LLMPlanner():
    def __init__(self):
        self.prompt = "You are a drone pilot. You are provided with a toolset that allows \
            you to move the drone or get visual information. The toolset contains both \
            high-level and low-level tools, please use high-level tools as much as possible. \
            Here is an example: input: \"Find an apple\", output: \"exec#high,find,apple exec#high,centering,apple\""