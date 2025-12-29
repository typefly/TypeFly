from .skill_item import SkillItem

class SkillSet():
    """
    The set of skills that supported by the robot.
    """
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
        del self.skills[name]
    
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
    