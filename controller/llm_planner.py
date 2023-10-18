import os, json, ast
from typing import Union

from skillset import SkillSet
from llm_wrapper import LLMWrapper
from vision_wrapper import VisionWrapper

class LLMPlanner():
    def __init__(self):
        self.llm = LLMWrapper()
        self.example_requests = [
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
                "response": ["loop#8#3", "if#query,'is there any animal?',=,true#1", "ret#true", "exec#turn_cw,45", "ret#false"],
                "explanation": "Loop 8 times, if the answer to the question is true, return true. Otherwise, turn 45 degrees counter-clockwise. If the loop is finished, return false."
            },
            {
                "query": {
                    "objects": ["chair", "laptop"],
                    "text": "[A] Find something edible, if you can't, then find something to drink."
                },
                "response": ["loop#8#4", "if#query,'is there anything edible?',=,true#2", "exec#approach", "ret#true", "exec#turn_cw,45",
                             "loop#8#4", "if#query,'is there anything drinkable?',=,true#2", "exec#approach", "ret#true", "exec#turn_cw,45",
                             "str#'no edible and drinkable item can be found'", "ret#false"],
                "explanation": "Loop 8 times, if the answer to the question is true, return true. Otherwise, turn 45 degrees counter-clockwise. If the loop is finished, return false."
            }
        ]

        # read prompt from txt
        with open("./assets/planning_prompt.txt", "r") as f:
            self.planning_prompt = f.read()

        with open("./assets/verification_prompt.txt", "r") as f:
            self.verification_prompt = f.read()

        with open("./assets/execution_prompt.txt", "r") as f:
            self.execution_prompt = f.read()

        with open("./assets/rules.txt", "r") as f:
            self.rules = f.read()

        with open("./assets/minispec_syntax.txt", "r") as f:
            self.minispec_syntax = f.read()

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def request_planning(self, task_description: str):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        prompt = self.planning_prompt.format(system_skill_description_high=self.high_level_skillset,
                                             system_skill_description_low=self.low_level_skillset,
                                             minispec_syntax=self.minispec_syntax,
                                             rules=self.rules,
                                             examples=self.example_requests,
                                             scene_description=self.vision_skill.get_obj_list(),
                                             task_description=task_description)
        print(f"> Planning request: {task_description}...")
        return ast.literal_eval(self.llm.request(prompt))

    def request_verification(self, prev_task_description: str, prev_task_response: str):
        prompt = self.verification_prompt.format(rules=self.rules,
                                                 scene_description=self.vision_skill.get_obj_list(),
                                                 task_description=prev_task_description,
                                                 plan=prev_task_response)
        print(f"> Verification request: {prev_task_description}...")
        return ast.literal_eval(self.llm.request(prompt))
    
    def request_execution(self, question: str) -> Union[bool, str]:
        def parse_value(s):
            # Check for boolean values
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            return s
        prompt = self.execution_prompt.format(scene_description=self.vision_skill.get_obj_list(), question=question)
        print(f"> Execution request: {question}...")
        return parse_value(self.llm.request(prompt))