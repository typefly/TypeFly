import sys
sys.path.append('..')
from typefly.platforms.virtual_robot_wrapper import VirtualRobotWrapper
from typefly.llm_controller import LLMController
from typefly.minispec_interpreter import MiniSpecProgram

llm_output = '```json\n{\n    \"thoughts\": \"The user instruction is to turn around, which typically means a 180-degree rotation. This can be achieved by using the turn_cw skill with 180 degrees.\",\n    \"<plan, robot1>\": \"mf(180);tp()\"\n}\n```'

controller = LLMController([], None)
robot = VirtualRobotWrapper(None, controller.controller_func)

program = MiniSpecProgram(robot)
print(program.parse(llm_output, True))
print(program.eval())