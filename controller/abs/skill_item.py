from abc import ABC, abstractmethod

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}:{self.arg_type.__name__}"

class SkillItem(ABC):
    abbr_dict = {}
    def generate_abbreviation(self, word):
        base_abbr = ''.join([part[0] for part in word.split('_')])
        abbr = base_abbr
        count = 1
        while abbr in self.abbr_dict:
            abbr = base_abbr + str(count)
            count += 1
        self.abbr_dict[abbr] = word
        return abbr

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_skill_description(self) -> str:
        pass

    @abstractmethod
    def get_argument(self) -> [SkillArg]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def execute(self, arg_list: [str]):
        pass