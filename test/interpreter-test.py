import sys
sys.path.append("..")
from controller.platforms.virtual_robot_wrapper import VirtualRobotWrapper
from controller.llm_controller import LLMController
from controller.minispec_interpreter import MiniSpecInterpreter, MiniSpecProgram

llm_output = '```json\n{\n    \"thoughts\": \"The user instruction is to turn around, which typically means a 180-degree rotation. This can be achieved by using the turn_cw skill with 180 degrees.\",\n    \"<plan, robot1>\": \"s(\'bottle\')\"\n}\n```'

controller = LLMController([], None)
robot = VirtualRobotWrapper(None, controller.system_skill_func)

program = MiniSpecProgram(robot)
print(program.parse(llm_output))
print(program.eval())