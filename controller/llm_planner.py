import os
import openai

from toolset import ToolSet

openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')
MODEL_NAME = "gpt-3.5-turbo-16k"

from llm_controller import LLMController

class LLMPlanner():
    def __init__(self, high_level_toolset: ToolSet, low_level_toolset: ToolSet):
        self.temperture = 0.0
        self.high_level_toolset = high_level_toolset
        self.low_level_toolset = low_level_toolset
        self.prompt = "You are a drone pilot. You are provided with a toolset that allows \
            you to move the drone or get visual information. The toolset contains both \
            high-level and low-level tools, please use high-level tools as much as possible. \
            High-level tools: {}, low-level tools: {}. \
            Here is an example: input: \"Find an apple\", output: \"exec#high,find,apple exec#high,centering,apple\"\
            Then this is the user input: {}, please generate the output command without explanation."
        
    def request(self):
        prompt = self.prompt.format(self.high_level_toolset, self.low_level_toolset, "turn around and go find a chair")
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperture,
        )
        print(response)
        return response

if __name__ == '__main__':
    controller = LLMController()
    planner = LLMPlanner(controller.high_level_toolset, controller.low_level_toolset)
    planner.request()
    