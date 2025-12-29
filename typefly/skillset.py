from enum import Enum
from typing import Optional
from .skill_item import SkillItem, SkillArg

class SkillSet():
    def __init__(self):
        self.skills: dict[str, SkillItem] = {}
    
    def get_skill(self, name: str) -> SkillItem:
        """Returns a SkillItem by its name."""
        skill = self.skills.get(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found.")
        return skill
    
    def add_skill(self, func: callable, description: str):
        """Adds a skill to the set."""
        self.skills[func.__name__.lower()] = SkillItem(func, description)
    
    def remove_skill(self, name: str):
        """Removes a SkillItem from the set by its name."""
        if name not in self.skills:
            raise ValueError(f"No skill found with the name '{name}'.")
        # remove skill by value
        del self.skills[name]

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
    
    def __repr__(self) -> str:
        string = ""
        for skill in self.skills.values():
            string += f"{skill}\n"
        return string
    
    @staticmethod
    def get_common_skillset(skills: list[callable]) -> 'SkillSet':
        skillset = SkillSet()
        for skill in skills:
            skillset.add_skill(skill[0], skill[1])
        return skillset
    