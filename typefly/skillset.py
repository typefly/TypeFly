from enum import Enum
from typing import Optional
from .skill_item import SkillItem, SkillArg, LowLevelSkillItem, HighLevelSkillItem

class SkillSet():
    def __init__(self):
        self.skills: dict[str, SkillItem] = {}
    
    def get_skill(self, name: str) -> SkillItem:
        """Returns a SkillItem by its name."""
        skill = self.skills.get(name)
        if skill is None:
            raise ValueError(f"Skill '{name}' not found.")
        return skill
    
    def add_skill(self, name: str, func: callable, description: str, args: list[SkillArg] = None):
        """Adds a skill to the set."""
        self.skills[name] = SkillItem(name, func, description, args)
    
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
        skillset.add_low_level_skill("object_dist", vision_skills[6], "Get object's dist (m)", args=[SkillArg("obj", str)])

        skillset.add_low_level_skill("log", other_skills[0], "Print text to user", args=[SkillArg("text", str)])
        skillset.add_low_level_skill("delay", other_skills[1], "Wait for seconds", args=[SkillArg("sec", float)])
        skillset.add_low_level_skill("re_plan", other_skills[2], "Trigger replanning")
        skillset.add_low_level_skill("probe", other_skills[3], "Query LLM for reasoning", args=[SkillArg("query", str)])

        return skillset
    