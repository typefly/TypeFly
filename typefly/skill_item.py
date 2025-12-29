from abc import ABC
import inspect

from typing import TYPE_CHECKING, Optional

SKILL_ARG_TYPE = int | float | str
PROBE_RET_TYPE = Optional[int | float | bool | str]

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}:{self.arg_type.__name__}"

class SkillItem(ABC):
    def __init__(self, func: callable, description: str):
        # Auto-inspect function to get name
        self._name = func.__name__.lower()
        
        self._description = description
        self._func = func  # Store the function so it can be called
        
        # Auto-inspect function signature to get arguments
        sig = inspect.signature(func)   
        self._args = []
        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter if present
            if param_name == 'self':
                continue
            
            # Require explicit type annotation
            if param.annotation == inspect.Parameter.empty:
                raise TypeError(
                    f"Function '{self._name}' parameter '{param_name}' must have an explicit type annotation. "
                    f"Example: def {self._name}({param_name}: int, ...)"
                )
            
            param_type = param.annotation
            self._args.append(SkillArg(param_name, param_type))
    
    def __call__(self, *args, **kwargs):
        """Make SkillItem callable by delegating to the stored function."""
        return self._func(*args, **kwargs)     

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def args(self) -> list[SkillArg]:
        return self._args
    
    def __repr__(self) -> str:
        return f"name: {self._name}, description: {self._description}, args: {[arg for arg in self._args]}"