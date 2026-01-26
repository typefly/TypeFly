import os
from dotenv import load_dotenv
import time

load_dotenv()

from typefly.llm_controller import LLMController
from typefly.robot_info import RobotInfo

print("="*70)
print("🚁 Janah - Testing UAV Movement Commands")
print("="*70)

robot_info = RobotInfo.from_dict({
    'robot_id': 'janah_movement_test',
    'robot_type': 'virtual',
    'extra': {'capture': 0}
})

controller = LLMController(robot_info)
controller.set_missing_child_info({
    'name': 'سارة',
    'age': 5,
    'clothing_color': 'فستان وردي',
    'last_location': 'مدينة الألعاب'
})

controller.start_controller()

# Test different movement commands in Arabic
commands = [
    "ارتفع 5 أمتار",
    "تحرك للأمام 10 أمتار",
    "دُر يميناً 90 درجة",
    "ابحث في دائرة حول الموقع الحالي",
]

for i, cmd in enumerate(commands, 1):
    print(f"\n[Test {i}/4] Command: {cmd}")
    controller.put_instruction(cmd)
    time.sleep(8)

print("\n" + "="*70)
print("✅ Movement test completed!")
print("="*70)

controller.stop_controller()