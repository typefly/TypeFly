import sys
sys.path.append("..")
from controller.llm_controller import LLMController

controller = LLMController()
try:
    controller.minispec_interpreter.execute("o,apple")
except Exception as e:
    print(e)