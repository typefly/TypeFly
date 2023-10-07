from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}: {self.arg_type.__name__}"

class SkillItem(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_comment(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def execute(self, args_str: str):
        pass

class SkillSetLevel(Enum):
        LOW = "low"
        HIGH = "high"

class SkillSet():
    def __init__(self, level = "low", lower_level_skillset: 'SkillSet' = None):
        self.skills = {}
        self.level = SkillSetLevel(level)
        self.lower_level_skillset = lower_level_skillset

    def get_skill(self, skill_name: str) -> Optional[SkillItem]:
        """Returns a SkillItem by its name."""
        if skill_name not in self.skills:
            return None
        return self.skills[skill_name]
    
    def add_skill(self, skill_item: SkillItem):
        """Adds a SkillItem to the set."""
        if skill_item.skill_name in self.skills:
            raise ValueError(f"A skill with the name '{skill_item.skill_name}' already exists.")
        if self.level == SkillSetLevel.HIGH and self.lower_level_skillset is not None:
            skill_item.set_low_level_skillset(self.lower_level_skillset)
        self.skills[skill_item.skill_name] = skill_item
    
    def remove_skill(self, skill_name: str):
        """Removes a SkillItem from the set by its name."""
        if skill_name not in self.skills:
            raise ValueError(f"No skill found with the name '{skill_name}'.")
        del self.skills[skill_name]
    
    def execute_skill(self, skill_name: str, args_str: str):
        """Executes a skill by its name with the provided arguments."""
        if skill_name not in self.skills:
            raise ValueError(f"No skill found with the name '{skill_name}'.")
        return self.skills[skill_name].execute(args_str.split(','))
    
    def __repr__(self) -> str:
        string = ""
        for skill in self.skills.values():
            string += f"{skill}\n"
        return string

class LowLevelSkillItem(SkillItem):
    def __init__(self, skill_name: str, skill_callable: callable = None,
                 comment: str = "", args: [SkillArg] = None):
        self.skill_name = skill_name
        self.skill_callable = skill_callable
        self.comment = comment
        self.args = args if args is not None else [SkillArg("object_name", str)]

    def get_name(self) -> str:
        return self.skill_name
    
    def get_comment(self) -> str:
        return self.comment
    
    def is_args_valid(self, args_str: str) -> bool:
        """Checks if the provided arguments are valid for the skill."""
        try:
            self.parse_args(args_str, allow_positional_args=True)
            return True
        except ValueError:
            return False

    def parse_args(self, args_str_list: [str], allow_positional_args: bool = False):
        """Parses the string of arguments and converts them to the expected types."""
        # Check the number of arguments
        if len(args_str_list) != len(self.args):
            raise ValueError(f"Expected {len(self.args)} arguments, but got {len(args_str_list)}.")
        
        parsed_args = []
        for i, arg in enumerate(args_str_list):
            # Allow positional arguments
            if arg.startswith('$') and allow_positional_args:
                parsed_args.append(arg)
                continue
            try:
                parsed_args.append(self.args[i].arg_type(arg.strip()))
            except ValueError as e:
                raise ValueError(f"Error parsing argument {i + 1}. Expected type {self.args[i].arg_type.__name__}, but got value '{arg.strip()}'. Original error: {e}")
        return parsed_args
    
    def execute(self, args_str_list: [str]):
        """Executes the skill with the provided arguments."""
        if callable(self.skill_callable):
            parsed_args = self.parse_args(args_str_list)
            return self.skill_callable(*parsed_args)
        else:
            raise ValueError(f"'{self.skill_callable}' is not a callable function.")

    def __repr__(self) -> str:
        return (f"Skill("
                f"name: {self.skill_name}, "
                f"args: {[arg for arg in self.args]}, "
                f"comment: {self.comment})")

class HighLevelSkillItem(SkillItem):
    def __init__(self, skill_name: str, skill_str_list: [str],
                 comment: str = ""):
        self.skill_name = skill_name
        self.skill_str_list = skill_str_list
        self.comment = comment
        self.low_level_skillset = None
        self.args = []

    def get_name(self) -> str:
        return self.skill_name
    
    def get_comment(self) -> str:
        return self.comment

    def set_low_level_skillset(self, low_level_skillset: SkillSet):
        self.low_level_skillset = low_level_skillset
        self.args = self.generate_arg_types()

    def generate_arg_types(self):
        """Parses the skill string, generate arg list and check \
            if the arguments are valid for the low-level skills."""
        args = []
        positional_args = []
        for skill_str in self.skill_str_list:
            for segment in skill_str.split("#"):
                split = segment.split(",")
                skill_name = split[0]
                skill = self.low_level_skillset.get_skill(skill_name)
                # append two list into arg_types and arg_names
                for i, arg_str in enumerate(split[1:]):
                    if arg_str.startswith("$"):
                        arg_instance = skill.args[i]
                        arg_index = int(arg_str[1:])
                        if arg_index not in positional_args:
                            positional_args.append(arg_index)
                            args.append(arg_instance)
                        elif args[positional_args.index(arg_index)].arg_type != arg_instance.arg_type:
                            raise ValueError(f"Argument {arg_index} is used twice with different types.")
        return args

    def execute(self, args_str_list: [str]):
        """Executes the skill with the provided arguments."""
        if self.low_level_skillset is None:
            raise ValueError("Low-level skillset is not set.")
        if len(args_str_list) != len(self.args):
            raise ValueError(f"Expected {len(self.args)} arguments, but got {len(args_str_list)}.")
        # replace all $1, $2, ... with segments
        skill_str_list = self.skill_str_list.copy()
        for index, skill_str in enumerate(skill_str_list):
            for i in range(0, len(args_str_list)):
                skill_str = skill_str.replace(f"${i + 1}", args_str_list[i])
            skill_str_list[index] = skill_str
        return skill_str_list

    def __repr__(self) -> str:
        return (f"Skill("
                f"name: {self.skill_name}, "
                f"skill_str_list: {self.skill_str_list}, "
                f"args: {[arg for arg in self.args]}, "
                f"comment: {self.comment})")

############################################
def main():
    def is_not_in_sight(obj: str):
        return f"Hello, {obj}!"

    def turn_left(deg: int):
        return deg
    # Create SkillItem objects
    low_level_skill_1 = LowLevelSkillItem("is_not_in_sight", is_not_in_sight)
    low_level_skill_2 = LowLevelSkillItem("turn_left", turn_left, args=[SkillArg("degree", int)])

    # Create a SkillSet and add the skills
    skillset = SkillSet()
    skillset.add_skill(low_level_skill_1)
    skillset.add_skill(low_level_skill_2)

    # Create a high level skill
    high_level_skill_1 = HighLevelSkillItem("find",
                                        skill_str_list=["loop#4", "if#is_not_in_sight,$1#2", "exec#turn_left,10", "skip#1", "break"],
                                        comment="turn around until find the object in sight")
    
    high_level_skill_2 = HighLevelSkillItem("orienting",
                                                           ["loop#4#7",
                                                            "if#is_not_in_sight,$1,>,0.6#1", "exec#turn_cw,15",
                                                            "if#is_not_in_sight,$1,<,0.4#1", "exec#turn_ccw,15",
                                                            "if#is_not_in_sight,$1,<,0.6#2", "if#is_not_in_sight,$1,>,0.4#1", "return#true", "return#false"],
                                                            "align the object to the center of the frame by rotating the drone")
    
    high_level_skillset = SkillSet(level="high", lower_level_skillset=skillset)
    high_level_skillset.add_skill(high_level_skill_1)
    high_level_skillset.add_skill(high_level_skill_2)

    print(skillset)
    print(high_level_skillset)

    # Execute skills by their names
    print(skillset.execute_skill("is_not_in_sight", "Alice"))
    print(skillset.execute_skill("turn_left", "5"))

if __name__ == "__main__":
    main()