import sys
sys.path.append('..')
from typefly.platforms.virtual_robot_wrapper import VirtualRobotWrapper
from typefly.llm_controller import LLMController
from typefly.minispec_interpreter import MiniSpecProgram
from typefly.robot_info import RobotInfo

llm_output = '```json\n{\n    \"thoughts\": \"The.\",\n    \"<plan, robot1>\": \"?True{'
info = {
    "robot_type": "virtual",
    "robot_id": "robot1",
    "extra": {
        "capture": 0,
    }
}
r_info = RobotInfo.from_dict(info)
controller = LLMController([r_info], None)

robot = VirtualRobotWrapper(r_info, controller.controller_func)

program = MiniSpecProgram({r_info: robot})
print(program.parse(llm_output, True))
print(program.parse("s('apple');", True))

print(program.statement.executable)
print(program.statement.eval())