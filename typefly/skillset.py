from enum import Enum
from typing import Optional
from .skill_item import SkillItem, SkillArg, LowLevelSkillItem, HighLevelSkillItem

class SkillSetLevel(Enum):
    LOW = "low"
    HIGH = "high"

class SkillSet():
    def __init__(self, level: SkillSetLevel = SkillSetLevel.LOW, lower_level_skillset: 'SkillSet' = None):
        self.skills: dict[str, SkillItem] = {}
        self.level = level
        self.lower_level_skillset = lower_level_skillset
        self.abbr_dict: dict[str, str] = {}

        if lower_level_skillset is not None:
            self.abbr_dict = {**lower_level_skillset.abbr_dict}
    
    def get_skill(self, name: str) -> Optional[SkillItem]:
        """Returns a SkillItem by its name or abbr."""
        skill = self.skills.get(name)
        if skill is None:
            skill = self.skills.get(self.abbr_dict.get(name, ''))
        return skill
    
    def add_low_level_skill(self, name: str, func: callable, description: str, args: list[SkillArg] = None):
        """Adds a LowLevelSkillItem to the set."""
        if self.level != SkillSetLevel.LOW:
            raise ValueError("Cannot add low-level skill to high-level skillset.")
        
        if name in self.skills:
            raise ValueError(f"A skill with the name '{name}' already exists.")
        
        abbr = self.generate_abbreviation(name)
        self.skills[name] = LowLevelSkillItem(name, abbr, func, description, args)

    def add_high_level_skill(self, name: str, definition: str, description: str):
        """Adds a HighLevelSkillItem to the set."""
        if self.level != SkillSetLevel.HIGH:
            raise ValueError("Cannot add high-level skill to low-level skillset.")
        
        if name in self.skills:
            raise ValueError(f"A skill with the name '{name}' already exists.")
        
        abbr = self.generate_abbreviation(name)
        self.skills[name] = HighLevelSkillItem(name, abbr, definition, description, [self.lower_level_skillset, self])
    
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
    def get_common_skillset(movement_skills: list[callable], vision_skills: list[callable], other_skills: list[callable]) -> 'SkillSet':
        skillset = SkillSet(level=SkillSetLevel.LOW)
        skillset.add_low_level_skill("move", movement_skills[0], "Move by (dx, dy) cm distance (dx: +forward/-backward, dy: +left/-right)", args=[SkillArg("dx", float), SkillArg("dy", float)])
        skillset.add_low_level_skill("rotate", movement_skills[1], "Rotate by a certain degree (deg: +left/-right)", args=[SkillArg("deg", float)])

        skillset.add_low_level_skill("is_visible", vision_skills[0], "Check if object is visible", args=[SkillArg("obj", str)])
        skillset.add_low_level_skill("object_x", vision_skills[1], "Get object's x position (0-1)", args=[SkillArg("obj", str)])
        skillset.add_low_level_skill("object_y", vision_skills[2], "Get object's y position (0-1)", args=[SkillArg("obj", str)])
        skillset.add_low_level_skill("object_width", vision_skills[3], "Get object's width (0-1)", args=[SkillArg("obj", str)])
        skillset.add_low_level_skill("object_height", vision_skills[4], "Get object's height (0-1)", args=[SkillArg("obj", str)])
        skillset.add_low_level_skill("take_picture", vision_skills[5], "Take a picture")

        skillset.add_low_level_skill("log", other_skills[0], "Print text to user", args=[SkillArg("text", str)])
        skillset.add_low_level_skill("delay", other_skills[1], "Wait for seconds", args=[SkillArg("sec", float)])
        skillset.add_low_level_skill("re_plan", other_skills[2], "Trigger replanning")
        skillset.add_low_level_skill("probe", other_skills[3], "Query LLM for reasoning", args=[SkillArg("query", str)])

        return skillset
    