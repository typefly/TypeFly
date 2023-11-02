import sys
sys.path.append("..")
from controller.llm_controller import LLMController
from controller.minispec_interpreter import MiniSpecInterpreter

controller = LLMController()
print(controller.low_level_skillset)
print(controller.high_level_skillset)

MiniSpecInterpreter.low_level_skillset = controller.low_level_skillset
MiniSpecInterpreter.high_level_skillset = controller.high_level_skillset
interpreter = MiniSpecInterpreter()

print(interpreter.execute("8{_1=q,'what's the edible target?';?_1!=False{o,_1;a;->True}tc,45}"))