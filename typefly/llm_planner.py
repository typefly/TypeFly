import os
from typing import Optional

from .llm_wrapper import LLMWrapper, ModelType
from .utils import print_t, CURRENT_PROJ_DIR, sanitize_prompt_text
from .robot_wrapper import RobotWrapper
from .robot_info import RobotInfo

def build_replan_feedback(error_message: Optional[list[str]] = None,
                          execution_history: Optional[list[str]] = None) -> str:
    """Build a corrective feedback block appended to the planning prompt on retry.

    Returns '' when there is no prior failure to report, so the first attempt's
    prompt is unchanged."""
    sections = []
    if error_message:
        reasons = "\n".join(f"- {m}" for m in error_message)
        sections.append(
            "Your previous plan was REJECTED before execution and must be fixed.\n"
            "Reason(s):\n" + reasons + "\n"
            "Return a corrected JSON plan. Plans may only call the listed skills "
            "with literal/variable arguments and simple control flow (if/else, "
            "bounded for-range, break). No imports, attribute access, arithmetic, "
            "while-loops, comprehensions, or f-string format specifiers."
        )
    if execution_history:
        history = "\n".join(f"- {h}" for h in execution_history)
        sections.append("Execution so far:\n" + history)
    if not sections:
        return ""
    return "\n\n# PREVIOUS ATTEMPT FEEDBACK\n" + "\n\n".join(sections) + "\n"


class LLMPlanner():
    def __init__(self, robot: RobotWrapper, model_type: ModelType = ModelType.GPT4O):
        self.llm = LLMWrapper()
        self.robot = robot
        self.model_type = model_type

        assets_path = os.path.join(CURRENT_PROJ_DIR, f"./assets")
        # read prompt from txt
        with open(os.path.join(assets_path, "prompt_plan.txt"), "r") as f:
            self.prompt_plan = f.read()
        with open(os.path.join(assets_path, "prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()
        with open(os.path.join(assets_path, "guidelines.txt"), "r") as f:
            self.guidelines = f.read()
        with open(os.path.join(assets_path, "example_plans.txt"), "r") as f:
            self.example_plans = f.read()

    def plan(self, user_instruction: str, error_message: Optional[list[str]]=None, execution_history: Optional[list[str]]=None):
        """
        Plan the user instruction using the LLM
        """
        # user_instruction is untrusted; flatten it (defense-in-depth). The real
        # safety boundary is the AST validator + PlanPolicy in plan_execution.
        prompt = self.prompt_plan.format(guidelines=self.guidelines,
                                         robot_skills=str(self.robot.skillset),
                                         example_plans=self.example_plans,
                                         scene_description=self.robot.get_obj_list_str(),
                                         user_instruction=sanitize_prompt_text(user_instruction))

        # On a retry, tell the model exactly why the last plan was rejected so it
        # can correct it instead of (likely) repeating the same invalid output.
        prompt += build_replan_feedback(error_message, execution_history)

        # json_mode: planning output must be a single JSON object.
        return self.llm.request(prompt, self.model_type, json_mode=True)

    def probe(self, query: str, robot_info: RobotInfo) -> str:
        """
        Probe the LLM for question $query based on the scene description
        """
        prompt = self.prompt_probe.format(scene_description=self.robot.get_obj_list_str(),
                                          query=sanitize_prompt_text(query))
        # No json_mode: probe returns free-form text (e.g. 'bottle', 'True', '2').
        return self.llm.request(prompt, self.model_type)