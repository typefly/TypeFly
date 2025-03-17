import os
from typing import Optional

from .llm_wrapper import LLMWrapper, ModelType
from .utils import print_t
from .robot_wrapper import RobotWrapper
from .robot_info import RobotInfo

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class LLMPlanner():
    def __init__(self, model_type: ModelType = ModelType.GPT4O):
        self.llm = LLMWrapper()
        self.model_type = model_type
        self.robot_list = []

        # read prompt from txt
        with open(os.path.join(CURRENT_DIR, f"./assets/prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/guidelines.txt"), "r") as f:
            self.guidelines = f.read()

        with open(os.path.join(CURRENT_DIR, f"./assets/example_plans.txt"), "r") as f:
            self.example_plans = f.read()

    def set_robot_dict(self, robot_dict: dict[RobotInfo, RobotWrapper]):
        self.robot_dict = robot_dict

    def plan(self, user_instruction: str, error_message: Optional[list[str]]=None, execution_history: Optional[list[str]]=None):
        robot_skills = ""
        scene_description = ""

        for info, robot in self.robot_dict.items():
            robot_skills += f"### {info.robot_id} ({info.get_robot_type(False)})\n"
            robot_skills += f"#### Low-level skills\n"
            robot_skills += str(robot.ll_skillset)
            if robot.hl_skillset is not None:
                robot_skills += f"\n#### High-level skills\n"
                robot_skills += str(robot.hl_skillset)

            scene_description += f"### {info.robot_id} ({info.get_robot_type(False)})\n"
            scene_description += robot.get_obj_list_str() + "\n"

        prompt = self.prompt_plan.format(guidelines=self.guidelines,
                                         robot_skills=robot_skills,
                                         example_plans=self.example_plans,
                                         scene_description=scene_description,
                                         user_instruction=user_instruction)

        return self.llm.request(prompt, self.model_type, stream=False)
    
    def probe(self, query: str, robot_info: RobotInfo) -> str:
        prompt = self.prompt_probe.format(scene_description=self.robot_dict[robot_info].get_obj_list_str(), query=query)
        print_t(f"[P] Execution request: {query}")
        return self.llm.request(prompt, self.model_type)