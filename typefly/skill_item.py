from abc import ABC, abstractmethod
from overrides import overrides
import re

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from skillset import SkillSet  # Import only for type checking

SKILL_ARG_TYPE = int | float | str
SKILL_RET_TYPE = Optional[int | float | bool | str]

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}:{self.arg_type.__name__}"

class SkillItem(ABC):
    def __init__(self, name: str, abbr: str, description: str):
        self._name = name
        self._description = description
        self._args = []
        self._abbr = abbr

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def args(self) -> list[SkillArg]:
        return self._args
    
    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, arg_list: list[SKILL_ARG_TYPE]) -> tuple[SKILL_RET_TYPE, bool]:
        pass

    def parse_args(self, args_str_list: list[SKILL_ARG_TYPE], allow_positional_args: bool = False):
        """Parses the string of arguments and converts them to the expected types."""
        # Check the number of arguments
        if len(args_str_list) != len(self.args):
            raise ValueError(f"Func {self.name} expected {len(self.args)} arguments, but got {len(args_str_list)}.")
        
        parsed_args = []
        for i, arg in enumerate(args_str_list):
            # if arg is not a string, skip parsing
            if not isinstance(arg, str):
                parsed_args.append(arg)
                continue
            # Allow positional arguments
            if arg.startswith('$') and allow_positional_args:
                parsed_args.append(arg)
                continue
            try:
                if self.args[i].arg_type == bool:
                    parsed_args.append(arg.strip().lower() == 'true')
                else:
                    parsed_args.append(self.args[i].arg_type(arg.strip()))
            except ValueError as e:
                raise ValueError(f"Error parsing argument {i + 1}. Expected type {self.args[i].arg_type.__name__}, but got value '{arg.strip()}'. Original error: {e}")
        return parsed_args
    
class LowLevelSkillItem(SkillItem):
    def __init__(self, name: str, abbr: str, func: callable, description: str, args: list[SkillArg] = None):
        super().__init__(name, abbr, description)
        self._callable = func
        self._args = args or []
    
    @overrides
    def execute(self, arg_list: list[SKILL_ARG_TYPE]) -> tuple[SKILL_RET_TYPE, bool]:
        """Executes the skill with the provided arguments."""
        if callable(self._callable):
            parsed_args = self.parse_args(arg_list)
            return self._callable(*parsed_args)
        else:
            raise ValueError(f"'{self._callable}' is not a callable function.")

    @overrides
    def __repr__(self) -> str:
        return (f"abbr: {self._abbr}, "
                f"name: {self._name}, "
                f"args: {[arg for arg in self._args]}, "
                f"description: {self._description}")

class HighLevelSkillItem(SkillItem):
    def __init__(self, name: str, abbr: str, definition: str, description: str, skill_set_list: list['SkillSet'] = None):
        super().__init__(name, abbr, description)
        self.definition = definition
        self.skill_set_list = skill_set_list or []
        self._args = self.generate_argument_list()

    @staticmethod
    def load_from_dict(skill_dict: dict) -> 'HighLevelSkillItem':
        return HighLevelSkillItem(skill_dict["name"], skill_dict["definition"], skill_dict["description"])

    def generate_argument_list(self) -> list[SkillArg]:
        # Extract all skill calls with their arguments from the code
        skill_calls = re.findall(r'(\w+)\(([^)]*)\)', self.definition)

        arg_types = {}

        for name, args in skill_calls:
            function_args = []
            args = [a.strip() for a in args.split(',')]
            if name == "int":
                function_args = [SkillArg("value", int)]
            elif name == "float":
                function_args = [SkillArg("value", float)]
            elif name == "str":
                function_args = [SkillArg("value", str)]
            else:
                skill = None
                for skill_set in self.skill_set_list:
                    skill = skill_set.get_skill(name)
                    if skill:
                        break

                if skill is None:
                    raise ValueError(f"Skill '{name}' not found in any skillset.")
                function_args = skill._args

            for i, arg in enumerate(args):
                if arg.startswith('$') and arg not in arg_types:
                    # Match the positional argument with its type from the function definition
                    arg_types[arg] = function_args[i]

        # Convert the mapped arguments to a user-friendly list in order of $position
        arg_types = dict(sorted(arg_types.items()))
        arg_list = [arg for arg in arg_types.values()]

        return arg_list

    @overrides
    def execute(self, arg_list: list[SKILL_ARG_TYPE]) -> tuple[SKILL_RET_TYPE, bool]:
        """Executes the skill with the provided arguments."""
        if len(self.skill_set_list) < 2:
            raise ValueError("Low-level skillset is not set.")
        if len(arg_list) != len(self._args):
            raise ValueError(f"Expected {len(self._args)} arguments, but got {len(arg_list)}.")
        # replace all $1, $2, ... with segments
        definition = self.definition
        for i in range(0, len(arg_list)):
            definition = definition.replace(f"${i + 1}", arg_list[i])
        return (definition, False)

    @overrides
    def __repr__(self) -> str:
        return (f"abbr: {self._abbr}, "
                f"name: {self._name}, "
                f"definition: {self.definition}, "
                f"args: {[arg for arg in self._args]}, "
                f"description: {self._description}")