import os, json, ast
import openai

from skillset import SkillSet
from llm_wrapper import LLMWrapper

class LLMPlanner():
    def __init__(self, llm: LLMWrapper, high_level_skillset: SkillSet, low_level_skillset: SkillSet):
        self.llm = llm
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.current_query = {
            "objects": ["lemon", "chair"],
            "text": "[Q] is there anything edible?"
        }
        self.example_queries = [
            {
                "query": {
                    "objects": [],
                    "text": "[A] find a chair"
                },
                "response": ["if#find,chair,=,True#2", "exec#orienting,chair", "exec#move_forward,50"],
                "explanation": "Chair is not in the list, so the planner first try to find a chair. If it is found, align and approach it."
            },
            {
                "query": {
                    "objects": ["chair", "laptop"],
                    "text": "[A] find a chair"
                },
                "response": ["exec#orienting,chair", "exec#move_forward,50"],
                "explanation": "Chair is in the list, just align and approach it."
            },
            {
                "query": {
                    "objects": ["apple", "chair", "laptop", "lemon"],
                    "text": "[Q] is there anything edible?"
                },
                "response": ["str#Yes, there is an apple and a lemon in sight."],
                "explanation": "This is a query problem, so the planner will return a string to answer the question."
            },
            {
                "query": {
                    "objects": ["apple", "chair", "laptop", "lemon"],
                    "text": "[A] goto the apple"
                },
                "response": ["exec#orienting,apple", "exec#move_forward,50"],
                "explanation": "The apple is in the list, the planner will align the apple to the center of the frame and move forward."
            },
            {
                "query": {
                    "objects": ["chair", "laptop"],
                    "text": "[A] find something edible"
                },
                "response": ["exec#find_edible_obj"],
                "explanation": "There is no edible object in the list, so the planner will find an edible object first."
            }
        ]

        # read prompt from txt
        with open("./assets/task_prompt.txt", "r") as f:
            self.task_prompt = f.read()

        with open("./assets/ending_prompt.txt", "r") as f:
            self.ending_prompt = f.read()

        with open("./assets/rules.txt", "r") as f:
            self.rules = f.read()

    def request_task(self, objects: [str], command: str):
        self.current_query["objects"] = objects
        # by default, the command is an action
        if not command.startswith("["):
            command = "[A] " + command
        self.current_query["text"] = command
        prompt = self.task_prompt.format(high_level_skills=self.high_level_skillset,
                                    low_level_skills=self.low_level_skillset,
                                    rules=self.rules,
                                    examples=self.example_queries,
                                    user_query=self.current_query)
        print(f"> Task query: {self.current_query}...")
        return ast.literal_eval(self.llm.query(prompt))
    
    def request_ending(self, objects: [str], prev_command: str):
        prompt = self.ending_prompt.format(high_level_skills=self.high_level_skillset,
                                    low_level_skills=self.low_level_skillset,
                                    rules=self.rules,
                                    examples=self.example_queries,
                                    user_query=self.current_query,
                                    response=prev_command,
                                    feedback=objects)
        print(f"> Ending query: {self.current_query}...")
        return ast.literal_eval(self.llm.query(prompt))