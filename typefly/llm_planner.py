# -*- coding: utf-8 -*-
import os
from typing import Optional

from .llm_wrapper import LLMWrapper, ModelType
from .utils import print_t, CURRENT_PROJ_DIR
from .robot_wrapper import RobotWrapper
from .robot_info import RobotInfo


class LLMPlanner():
    def __init__(self, robot: RobotWrapper, model_type: ModelType = ModelType.GPT4O):
        self.llm = LLMWrapper()
        self.robot = robot
        self.model_type = model_type

        assets_path = os.path.join(CURRENT_PROJ_DIR, "./assets")
        with open(os.path.join(assets_path, "prompt_plan.txt"), "r", encoding="utf-8") as f:
            self.prompt_plan = f.read()
        with open(os.path.join(assets_path, "prompt_probe.txt"), "r", encoding="utf-8") as f:
            self.prompt_probe = f.read()
        with open(os.path.join(assets_path, "guidelines.txt"), "r", encoding="utf-8") as f:
            self.guidelines = f.read()
        with open(os.path.join(assets_path, "example_plans.txt"), "r", encoding="utf-8") as f:
            self.example_plans = f.read()

    def plan(self,
             user_instruction: str,
             error_message: Optional[list] = None,
             execution_history: Optional[list] = None,
             missing_child_info: Optional[dict] = None) -> str:
        """
        Plan the user instruction using the LLM.
        ✅ FIX #6: يمرر error_message للـ prompt عند الـ replan
        """
        # معلومات الطفل المفقود
        if missing_child_info and missing_child_info.get('name'):
            child_info_str = (
                f"Name: {missing_child_info.get('name', 'N/A')}, "
                f"Age: {missing_child_info.get('age', 'N/A')}, "
                f"Clothing: {missing_child_info.get('clothing_color', 'N/A')}, "
                f"Last seen: {missing_child_info.get('last_location', 'N/A')}, "
                f"Description: {missing_child_info.get('description', 'N/A')}"
            )
        else:
            child_info_str = "No child information provided yet"

        # ✅ FIX #6: أضف errors للـ prompt
        error_context = ""
        if error_message:
            errors_str = "\n".join([f"- {e}" for e in error_message])
            error_context = f"\n\n# PREVIOUS ERRORS - DO NOT REPEAT\nThe last plan had these errors, fix them:\n{errors_str}"

        prompt = self.prompt_plan.format(
            missing_child_info=child_info_str,
            guidelines=self.guidelines,
            robot_skills=str(self.robot.skillset),
            example_plans=self.example_plans,
            scene_description=self.robot.get_obj_list_str(),
            user_instruction=user_instruction
        ) + error_context

        return self.llm.request(prompt, self.model_type)

    def probe(self, query: str, robot_info: RobotInfo) -> str:
        prompt = self.prompt_probe.format(
            scene_description=self.robot.get_obj_list_str(),
            query=query
        )
        return self.llm.request(prompt, self.model_type)