from abc import ABC, abstractmethod
from typing import List, Union, Tuple

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}:{self.arg_type.__name__}"

class SkillItem(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_skill_description(self) -> str:
        pass

    @abstractmethod
    def get_argument(self) -> List[SkillArg]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def execute(self, arg_list: List[Union[int, float, str]]) -> Tuple[Union[int, float, bool, str], bool]:
        pass

    abbr_dict = {}
    def generate_abbreviation(self, word):
        split = word.split('_')
        abbr = ''.join([part[0] for part in split])[0:2]

        if abbr not in self.abbr_dict:
            self.abbr_dict[abbr] = word
            return abbr
        
        split = ''.join([part for part in split])[1:]

        count = 0
        while abbr in self.abbr_dict:
            abbr = abbr[0] + split[count]
            count += 1

        self.abbr_dict[abbr] = word
        return abbr

    def parse_args(self, args_str_list: List[Union[int, float, str]], allow_positional_args: bool = False):
        """Parses the string of arguments and converts them to the expected types."""
        # Check the number of arguments
        if len(args_str_list) != len(self.args):
            raise ValueError(f"Expected {len(self.args)} arguments, but got {len(args_str_list)}.")
        
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
                parsed_args.append(self.args[i].arg_type(arg.strip()))
            except ValueError as e:
                raise ValueError(f"Error parsing argument {i + 1}. Expected type {self.args[i].arg_type.__name__}, but got value '{arg.strip()}'. Original error: {e}")
        return parsed_args