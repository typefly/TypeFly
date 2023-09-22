from enum import Enum
from abc import ABC, abstractmethod

class ToolArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self):
        return f"{self.arg_name}: {self.arg_type.__name__}"

class ToolItem(ABC):
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

class ToolSetLevel(Enum):
        LOW = "low"
        HIGH = "high"

class ToolSet():
    def __init__(self, level = "low", lower_level_toolset: 'ToolSet' = None):
        self.tools = {}
        self.level = ToolSetLevel(level)
        self.lower_level_toolset = lower_level_toolset

    def get_tool(self, tool_name: str) -> ToolItem:
        """Returns a ToolItem by its name."""
        if tool_name not in self.tools:
            raise ValueError(f"No tool found with the name '{tool_name}'.")
        return self.tools[tool_name]
    
    def add_tool(self, tool_item: ToolItem):
        """Adds a ToolItem to the set."""
        if tool_item.tool_name in self.tools:
            raise ValueError(f"A tool with the name '{tool_item.tool_name}' already exists.")
        if self.level == ToolSetLevel.HIGH and self.lower_level_toolset is not None:
            tool_item.set_low_level_toolset(self.lower_level_toolset)
        self.tools[tool_item.tool_name] = tool_item
    
    def remove_tool(self, tool_name: str):
        """Removes a ToolItem from the set by its name."""
        if tool_name not in self.tools:
            raise ValueError(f"No tool found with the name '{tool_name}'.")
        del self.tools[tool_name]
    
    def execute_tool(self, tool_name: str, args_str: str):
        """Executes a tool by its name with the provided arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"No tool found with the name '{tool_name}'.")
        return self.tools[tool_name].execute(args_str.split(','))
    
    def __repr__(self) -> str:
        return (f"ToolSet(\n"
                f"  level: {self.level.value},\n"
                f"  tools: {[tool for tool in self.tools.values()]},\n"
                f")")

class LowLevelToolItem(ToolItem):
    def __init__(self, tool_name: str, tool_callable: callable = None,
                 comment: str = "", args: [ToolArg] = None):
        self.tool_name = tool_name
        self.tool_callable = tool_callable
        self.comment = comment
        self.args = args if args is not None else [ToolArg("object_name", str)]

    def get_name(self) -> str:
        return self.tool_name
    
    def get_comment(self) -> str:
        return self.comment
    
    def is_args_valid(self, args_str: str) -> bool:
        """Checks if the provided arguments are valid for the tool."""
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
        """Executes the tool with the provided arguments."""
        if callable(self.tool_callable):
            parsed_args = self.parse_args(args_str_list)
            return self.tool_callable(*parsed_args)
        else:
            raise ValueError(f"'{self.tool_callable}' is not a callable function.")

    def __repr__(self) -> str:
        return (f"ToolItem(\n"
                f"  name: {self.tool_name},\n"
                f"  args: {[arg for arg in self.args]},\n"
                f"  comment: {self.comment},\n"
                f")")

class HighLevelToolItem(ToolItem):
    def __init__(self, tool_name: str, tool_str: str = None,
                 comment: str = ""):
        self.tool_name = tool_name
        self.tool_str = tool_str
        self.comment = comment
        self.low_level_toolset = None
        self.args = []

    def get_name(self) -> str:
        return self.tool_name
    
    def get_comment(self) -> str:
        return self.comment

    def set_low_level_toolset(self, low_level_toolset: ToolSet):
        self.low_level_toolset = low_level_toolset
        self.args = self.generate_arg_types()

    def generate_arg_types(self):
        """Parses the tool string, generate arg list and check \
            if the arguments are valid for the low-level tools."""
        args = []
        positional_args = []
        for section in self.tool_str.split():
            segments = section.split("#")
            for segment in segments:
                if segment.startswith("low,"):
                    split = segment.split(",")
                    tool_name = split[1]
                    tool = self.low_level_toolset.get_tool(tool_name)
                    # append two list into arg_types and arg_names
                    for i, arg_str in enumerate(split[2:]):
                        if arg_str.startswith("$"):
                            arg_instance = tool.args[i]
                            arg_index = int(arg_str[1:])
                            if arg_index not in positional_args:
                                positional_args.append(arg_index)
                                args.append(arg_instance)
                            elif args[positional_args.index(arg_index)].arg_type != arg_instance.arg_type:
                                raise ValueError(f"Argument {arg_index} is used twice with different types.")
        return args
    
    def execute_tool_command(self, tool_command) -> bool:
        split = tool_command.split(",")
        level = split[0]
        tool_name = split[1]
        if level == 'low':
            tool = self.low_level_toolset.get_tool(tool_name)
            print(f'> > exec low-level tool: {tool}, {split[2:]}')
            return tool.execute(split[2:])
        return False

    def execute(self, args_str_list: [str]):
        """Executes the tool with the provided arguments."""
        if self.low_level_toolset is None:
            raise ValueError("Low-level toolset is not set.")
        if len(args_str_list) != len(self.args):
            raise ValueError(f"Expected {len(self.args)} arguments, but got {len(args_str_list)}.")
        # replace all $1, $2, ... with segments
        tool_str = self.tool_str
        for i in range(0, len(args_str_list)):
            tool_str = tool_str.replace(f"${i + 1}", args_str_list[i])
        return tool_str

    def __repr__(self) -> str:
        return (f"ToolItem(\n"
                f"  name: {self.tool_name},\n"
                f"  tool_str: {self.tool_str},\n"
                f"  args: {[arg for arg in self.args]},\n"
                f"  comment: {self.comment},\n"
                f")")

############################################
def main():
    def is_not_in_sight(obj: str):
        return f"Hello, {obj}!"

    def turn_left(deg: int):
        return deg
    # Create ToolItem objects
    low_level_tool_1 = LowLevelToolItem("is_not_in_sight", is_not_in_sight)
    low_level_tool_2 = LowLevelToolItem("turn_left", turn_left, args=[ToolArg("degree", int)])

    # Create a ToolSet and add the tools
    toolset = ToolSet()
    toolset.add_tool(low_level_tool_1)
    toolset.add_tool(low_level_tool_2)

    # Create a high level tool
    high_level_tool_1 = HighLevelToolItem("find",
                                        tool_str="loop#4 if#low,is_not_in_sight,$1#2 exec#low,turn_left,10 skip#1 break",
                                        comment="turn around until find the object in sight")
    
    high_level_toolset = ToolSet(level="high", lower_level_toolset=toolset)
    high_level_toolset.add_tool(high_level_tool_1)

    print(toolset)
    print(high_level_toolset)

    # Execute tools by their names
    print(toolset.execute_tool("is_not_in_sight", "Alice"))
    print(toolset.execute_tool("turn_left", "5"))

if __name__ == "__main__":
    main()