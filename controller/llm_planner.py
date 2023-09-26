import os, json, ast
import openai

from toolset import ToolSet

openai.organization = "org-sAnQwPNnbSrHg1XyR4QYALf7"
openai.api_key = os.environ.get('OPENAI_API_KEY')
# MODEL_NAME = "gpt-3.5-turbo-16k"
MODEL_NAME = "gpt-4"

class LLMPlanner():
    def __init__(self, high_level_toolset: ToolSet, low_level_toolset: ToolSet):
        self.temperture = 0.05
        self.high_level_toolset = high_level_toolset
        self.low_level_toolset = low_level_toolset
        self.current_query = {
            "objects": ["lemon", "chair"],
            "command": "[Q] is there anything edible?"
        }

        self.example_queries = [
            {
                "input": {
                    "objects": [],
                    "command": "[A] find a chair"
                },
                "output": ["exec#high,scan,chair", "exec#high,align,chair"],
                "explanation": "Since there is no object in the list, the planner will scan the environment to find the object."
            },
            {
                "input": {
                    "objects": ["apple", "chair", "laptop", "lemon"],
                    "command": "[Q] is there anything edible?"
                },
                "output": ["str#Yes, there is an apple and a lemon in sight."],
                "explanation": "This is a query problem, so the planner will return a string to answer the question."
            },
            {
                "input": {
                    "objects": ["apple", "chair", "laptop", "lemon"],
                    "command": "[A] goto the apple"
                },
                "output": ["exec#high,align,apple", "exec#low,move_forward,50"],
                "explanation": "Since the apple is in the list, the planner will align the apple to the center of the frame and move forward."
            },
            {
                "input": {
                    "objects": ["chair", "laptop"],
                    "command": "[A] find something edible"
                },
                "output": ["exec#low,turn_ccw,45", "redo"],
                "explanation": "Since there is no edible object in the list, the planner will turn around and redo this query."
            }
        ]

        # read prompt from txt
        with open("./assets/prompt.txt", "r") as f:
            self.prompt = f.read()

    def request(self, objects: [str], command: str):
        self.current_query["objects"] = objects
        self.current_query["command"] = command
        prompt = self.prompt.format(high_level_tools=self.high_level_toolset,
                                    low_level_tools=self.low_level_toolset,
                                    examples=self.example_queries,
                                    user_command=self.current_query)
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperture,
        )

        # save the message in a txt
        with open("./assets/chat_log.txt", "a") as f:
            f.write(prompt + "\n##################################\n")
            f.write(json.dumps(response) + "\n##################################\n")

        return ast.literal_eval(response["choices"][0]["message"]["content"])