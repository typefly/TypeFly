import os, re
from typing import Union

from .skillset import SkillSet
from .llm_wrapper import LLMWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t, evaluate_value

current_directory = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self):
        self.llm = LLMWrapper()

        # read prompt from txt
        with open(os.path.join(current_directory, "./assets/planning_prompt.txt"), "r") as f:
            self.planning_prompt = f.read()

        with open(os.path.join(current_directory, "./assets/verification_prompt.txt"), "r") as f:
            self.verification_prompt = f.read()

        with open(os.path.join(current_directory, "./assets/execution_prompt.txt"), "r") as f:
            self.execution_prompt = f.read()

        with open(os.path.join(current_directory, "./assets/rules.txt"), "r") as f:
            self.rules = f.read()

        with open(os.path.join(current_directory, "./assets/minispec_syntax.txt"), "r") as f:
            self.minispec_syntax = f.read()

        with open(os.path.join(current_directory, "./assets/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
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
                                             plan_examples=self.plan_examples,
                                             scene_description=self.vision_skill.get_obj_list(),
                                             task_description=task_description)
        print_t(f"[P] Planning request: {task_description}")
        return self.llm.request(prompt)

    def request_verification(self, prev_task_description: str, prev_task_response: str):
        prompt = self.verification_prompt.format(rules=self.rules,
                                                 scene_description=self.vision_skill.get_obj_list(),
                                                 task_description=prev_task_description,
                                                 response=prev_task_response)
        print_t(f"[P] Verification request: {prev_task_description}")
        return self.llm.request(prompt)
    
    def request_execution(self, question: str) -> Union[bool, str, int, float]:
        prompt = self.execution_prompt.format(scene_description=self.vision_skill.get_obj_list(), question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt))