import os, ast
from typing import Optional

from .skillset import SkillSet
from .llm_wrapper import LLMWrapper, ModelType
from .vision_skill_wrapper import VisionSkillWrapper
from .utils import print_t
from .minispec_interpreter import MiniSpecValueType, evaluate_value
from .abs.robot_wrapper import RobotType

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self, robot_type: RobotType):
        self.llm = LLMWrapper()
        self.model_type = ModelType.GPT4

        type_folder_name = 'tello'
        if robot_type == RobotType.GO2:
            type_folder_name = 'gear'

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/guides.txt"), "r") as f:
            self.guides = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/{type_folder_name}/plan_examples.txt"), "r") as f:
            self.plan_examples = f.read()

    def init(self, high_level_skillset: SkillSet, low_level_skillset: SkillSet, vision_skill: VisionSkillWrapper):
        self.high_level_skillset = high_level_skillset
        self.low_level_skillset = low_level_skillset
        self.vision_skill = vision_skill

    def plan(self, task_description: str, scene_description: Optional[str] = None, error_message: Optional[str] = None, execution_history: Optional[str] = None):
        # by default, the task_description is an action
        if not task_description.startswith("["):
            task_description = "[A] " + task_description

        if scene_description is None:
            scene_description = self.vision_skill.get_obj_list()
        prompt = self.prompt_plan.format(system_skill_description_high=self.high_level_skillset,
                                             system_skill_description_low=self.low_level_skillset,
                                             guides=self.guides,
                                             plan_examples=self.plan_examples,
                                             error_message=error_message,
                                             scene_description=scene_description,
                                             task_description=task_description,
                                             execution_history=execution_history)
        print_t(f"[P] Planning request: {task_description}")
        return self.llm.request(prompt, self.model_type, stream=False)
    
    def probe(self, question: str) -> MiniSpecValueType:
        prompt = self.prompt_probe.format(scene_description=self.vision_skill.get_obj_list(), question=question)
        print_t(f"[P] Execution request: {question}")
        return evaluate_value(self.llm.request(prompt, self.model_type)), False