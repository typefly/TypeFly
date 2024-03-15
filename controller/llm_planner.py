import os, re

from .skillset import SkillSet
from .llm_wrapper import LLMWrapper
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self):
        self.llm = LLMWrapper()

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, "./assets/planning_prompt.txt"), "r") as f:
            self.planning_prompt = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/query_prompt.txt"), "r") as f:
            self.query_prompt = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/minispec_syntax.txt"), "r") as f:
            self.minispec_syntax = f.read()

        with open(os.path.join(CURRENT_DIR, "./assets/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def request_planning(self, task_description: str, scene_description: str = None, error_message: str = None, previous_response: str = None, execution_status: str = None):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        if scene_description is None:
            scene_description = self.vision_skill.get_obj_list()
        prompt = self.planning_prompt.format(system_skill_description_high=self.high_level_skillset,
                                             system_skill_description_low=self.low_level_skillset,
                                            #  minispec_syntax=self.minispec_syntax,
                                             guides=self.guides,
                                             plan_examples=self.plan_examples,
                                             error_message=error_message,
                                             scene_description=scene_description,
                                             task_description=task_description,
                                             previous_response=previous_response,
                                             execution_status=execution_status)
        print_t(f"[P] Planning request: {task_description}")
        return self.llm.request(prompt)
    
    def request_execution(self, question: str) -> MiniSpecValueType:
        prompt = self.query_prompt.format(scene_description=self.vision_skill.get_obj_list(), question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt)), False