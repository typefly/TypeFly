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

        with open(os.path.join(current_directory, "./assets/query_prompt.txt"), "r") as f:
            self.query_prompt = f.read()

        with open(os.path.join(current_directory, "./assets/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(current_directory, "./assets/minispec_syntax.txt"), "r") as f:
            self.minispec_syntax = f.read()

        with open(os.path.join(current_directory, "./assets/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def request_planning(self, task_description: str, scene_description: str = None, error_message: str = None):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        if scene_description is None:
            scene_description = self.vision_skill.get_obj_list()
        prompt = self.planning_prompt.format(system_skill_description_high=self.high_level_skillset,
                                             system_skill_description_low=self.low_level_skillset,
                                             minispec_syntax=self.minispec_syntax,
                                             guides=self.guides,
                                             plan_examples=self.plan_examples,
                                             error_message=error_message,
                                             scene_description=scene_description,
                                             task_description=task_description)
        print_t(f"[P] Planning request: {task_description}")
        return self.llm.request(prompt)
    
    def request_execution(self, question: str) -> Union[bool, str, int, float]:
        prompt = self.query_prompt.format(scene_description=self.vision_skill.get_obj_list(), question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt))