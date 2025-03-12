import re
from enum import Enum
from typing import Optional, Union
from .skill_item import SkillItem, SkillArg

class SkillSetLevel(Enum):
    LOW = "low"
    HIGH = "high"

class SkillSet():
    def __init__(self, level: SkillSetLevel = SkillSetLevel.LOW, lower_level_skillset: 'SkillSet' = None):
        self.skills = {}
        self.level = level
        self.lower_level_skillset = lower_level_skillset
    
    def get_skill(self, name: str) -> Optional[SkillItem]:
        """Returns a SkillItem by its name or abbr."""
        skill = None
        if name in self.skills:
            skill = self.skills[name]
        elif name in SkillItem.abbr_dict:
            skill = self.skills.get(SkillItem.abbr_dict[name])
        return skill
    
    def add_skill(self, skill_item: SkillItem):
        """Adds a SkillItem to the set."""
        if skill_item.name in self.skills:
            raise ValueError(f"A skill with the name '{skill_item.name}' already exists.")
        # Set the low-level skillset for high-level skills
        if self.level == SkillSetLevel.HIGH and isinstance(skill_item, HighLevelSkillItem):
            if self.lower_level_skillset is not None:
                skill_item.set_skillset(self.lower_level_skillset, self)
            else:
                raise ValueError("Low-level skillset is not set.")

        self.skills[skill_item.name] = skill_item
    
    def remove_skill(self, name: str):
        """Removes a SkillItem from the set by its name."""
        if name not in self.skills:
            raise ValueError(f"No skill found with the name '{name}'.")
        # remove skill by value
        del self.skills[name]
    
    def __repr__(self) -> str:
        string = ""
        for skill in self.skills.values():
            string += f"{skill}\n"
        return string
    
    @staticmethod
    def get_common_skillset(movement_skills: list[callable], vision_skills: list[callable], system_skills: list[callable]) -> 'SkillSet':
        skillset = SkillSet(level=SkillSetLevel.LOW)
        skillset.add_skill(LowLevelSkillItem("move_forward", movement_skills[0], "Move forward by distance", args=[SkillArg("dist", int)]))
        skillset.add_skill(LowLevelSkillItem("move_backward", movement_skills[1], "Move backward by distance", args=[SkillArg("dist", int)]))
        skillset.add_skill(LowLevelSkillItem("move_left", movement_skills[2], "Move left by distance", args=[SkillArg("dist", int)]))
        skillset.add_skill(LowLevelSkillItem("move_right", movement_skills[3], "Move right by distance", args=[SkillArg("dist", int)]))
        skillset.add_skill(LowLevelSkillItem("turn_cw", movement_skills[4], "Rotate clockwise/right by degrees", args=[SkillArg("deg", int)]))
        skillset.add_skill(LowLevelSkillItem("turn_ccw", movement_skills[5], "Rotate counterclockwise/left by degrees", args=[SkillArg("deg", int)]))

        skillset.add_skill(LowLevelSkillItem("is_visible", vision_skills[0], "Check if object is visible", args=[SkillArg("obj", str)]))
        skillset.add_skill(LowLevelSkillItem("object_x", vision_skills[1], "Get object's x position (0-1)", args=[SkillArg("obj", str)]))
        skillset.add_skill(LowLevelSkillItem("object_y", vision_skills[2], "Get object's y position (0-1)", args=[SkillArg("obj", str)]))
        skillset.add_skill(LowLevelSkillItem("object_width", vision_skills[3], "Get object's width (0-1)", args=[SkillArg("obj", str)]))
        skillset.add_skill(LowLevelSkillItem("object_height", vision_skills[4], "Get object's height (0-1)", args=[SkillArg("obj", str)]))

        skillset.add_skill(LowLevelSkillItem("log", system_skills[0], "Print text to user", args=[SkillArg("text", str)]))
        skillset.add_skill(LowLevelSkillItem("delay", system_skills[1], "Wait for seconds", args=[SkillArg("sec", float)]))
        skillset.add_skill(LowLevelSkillItem("take_picture", system_skills[2], "Take a picture"))
        skillset.add_skill(LowLevelSkillItem("re_plan", system_skills[3], "Trigger replanning"))
        skillset.add_skill(LowLevelSkillItem("probe", system_skills[4], "Query LLM for reasoning", args=[SkillArg("query", str)]))

        return skillset
        
class LowLevelSkillItem(SkillItem):
    def __init__(self, name: str, func: callable, description: str, args: list[SkillArg] = None):
        super().__init__(name, description)
        self._callable = func
        self._args = args or []

        self.abbr = self.generate_abbreviation(name)
        self.abbr_dict[self.abbr] = name
    
    def execute(self, arg_list: list[Union[int, float, str]]):
        """Executes the skill with the provided arguments."""
        if callable(self._callable):
            parsed_args = self.parse_args(arg_list)
            return self._callable(*parsed_args)
        else:
            raise ValueError(f"'{self._callable}' is not a callable function.")

    def __repr__(self) -> str:
        return (f"abbr:{self.abbr},"
                f"name:{self._name},"
                f"args:{[arg for arg in self._args]},"
                f"description:{self._description}")

class HighLevelSkillItem(SkillItem):
    def __init__(self, name: str, definition: str, description: str):
        super().__init__(name, description)
        self._definition = definition

        self.abbr = self.generate_abbreviation(name)
        self.abbr_dict[self.abbr] = name
        self.low_level_skillset = None

    def load_from_dict(skill_dict: dict):
        return HighLevelSkillItem(skill_dict["name"], skill_dict["definition"], skill_dict["description"])

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def args(self) -> list[SkillArg]:
        return self._args

    def set_skillset(self, low_level_skillset: SkillSet, high_level_skillset: SkillSet):
        self.low_level_skillset = low_level_skillset
        self.high_level_skillset = high_level_skillset
        self._args = self.generate_argument_list()

    def generate_argument_list(self) -> list[SkillArg]:
        # Extract all skill calls with their arguments from the code
        skill_calls = re.findall(r'(\w+)\(([^)]*)\)', self._definition)

        arg_types = {}

        for skill_name, args in skill_calls:
            args = [a.strip() for a in args.split(',')]
            if skill_name == "int":
                function_args = [SkillArg("value", int)]
            elif skill_name == "float":
                function_args = [SkillArg("value", float)]
            elif skill_name == "str":
                function_args = [SkillArg("value", str)]
            else:
                skill = self.low_level_skillset.get_skill(skill_name)
                if skill is None:
                    skill = self.high_level_skillset.get_skill(skill_name)

                if skill is None:
                    raise ValueError(f"Skill '{skill_name}' not found in the low-level or high-level skillset.")

                function_args = skill._args
            for i, arg in enumerate(args):
                if arg.startswith('$') and arg not in arg_types:
                    # Match the positional argument with its type from the function definition
                    arg_types[arg] = function_args[i]

        # Convert the mapped arguments to a user-friendly list in order of $position
        arg_types = dict(sorted(arg_types.items()))
        arg_list = [arg for arg in arg_types.values()]

        return arg_list

    def execute(self, arg_list: list[Union[int, float, str]]):
        """Executes the skill with the provided arguments."""
        if self.low_level_skillset is None:
            raise ValueError("Low-level skillset is not set.")
        if len(arg_list) != len(self._args):
            raise ValueError(f"Expected {len(self._args)} arguments, but got {len(arg_list)}.")
        # replace all $1, $2, ... with segments
        definition = self._definition
        for i in range(0, len(arg_list)):
            definition = definition.replace(f"${i + 1}", arg_list[i])
        return definition

    def __repr__(self) -> str:
        return (f"abbr:{self.abbr},"
                f"name:{self._name},"
                f"definition:{self._definition},"
                f"args:{[arg for arg in self._args]},"
                f"description:{self._description}")