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
                "response": ["if#scan,chair,=,True#2", "exec#orienting,chair", "exec#approach,chair"],
                "explanation": "Chair is not in the list, so the planner first try to find a chair. If it is found, align and approach it."
            },
            {
                "query": {
                    "objects": ["chair", "laptop"],
                    "text": "[A] find a chair"
                },
                "response": ["exec#orienting,chair", "exec#approach,chair"],
                "explanation": "Chair is in the list, just align and approach it."
            },
            {
                "query": {
                    "objects": ["apple", "chair", "laptop", "lemon"],
                    "text": "[Q] is there anything edible?"
                },
                "response": ["str#'Yes, there is an apple and a lemon in sight.'"],
                "explanation": "This is a query problem, so the planner will return a string to answer the question."
            },
            {
                "query": {
                    "objects": ["laptop", "lemon"],
                    "text": "[Q] is there anything edible behind you?"
                },
                "response": ["exec#turn_ccw,180", "str#query,'is there anything edible?'"],
                "explanation": "This is a query problem but requires the drone to turn around first. So the planner will first turn around and then print the query result."
            },
            {
                "query": {
                    "objects": ["apple", "chair", "laptop", "lemon"],
                    "text": "[A] goto the apple"
                },
                "response": ["exec#orienting,apple", "exec#approach,apple"],
                "explanation": "The apple is in the list, the planner will align the apple to the center of the frame and move forward."
            },
            {
                "query": {
                    "objects": ["chair", "laptop"],
                    "text": "[A] Turn around until you can see some animal."
                },
                "response": ["loop#8#3", "if#query,'is there any animal?',=,True#1", "return#true", "exec#turn_cw,45", "return#false"],
                "explanation": "Loop 8 times, if the answer to the question is true, return true. Otherwise, turn 45 degrees counter-clockwise. If the loop is finished, return false."
            },
            {
                "query": {
                    "objects": ["chair", "laptop"],
                    "text": "[A] Find something edible, if you can't, then find something to drink."
                },
                "response": ["loop#8#4", "if#query,'is there anything edible?',=,True#2", "exec#approach", "return#true", "exec#turn_cw,45",
                             "loop#8#4", "if#query,'is there anything drinkable?',=,True#2", "exec#approach", "return#true", "exec#turn_cw,45",
                             "str#'no edible and drinkable item can be found'", "return#false"],
                "explanation": "Loop 8 times, if the answer to the question is true, return true. Otherwise, turn 45 degrees counter-clockwise. If the loop is finished, return false."
            }
        ]

        # read prompt from txt
        with open("./assets/task_prompt.txt", "r") as f:
            self.task_prompt = f.read()

        with open("./assets/ending_prompt.txt", "r") as f:
            self.ending_prompt = f.read()

        with open("./assets/rules.txt", "r") as f:
            self.rules = f.read()

        with open("./assets/skill_design.txt", "r") as f:
            self.skill_design = f.read()

    def request_task(self, objects: [str], command: str):
        self.current_query["objects"] = objects
        # by default, the command is an action
        if not command.startswith("["):
            command = "[A] " + command
        self.current_query["text"] = command
        prompt = self.task_prompt.format(high_level_skills=self.high_level_skillset,
                                    low_level_skills=self.low_level_skillset,
                                    skill_design=self.skill_design,
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