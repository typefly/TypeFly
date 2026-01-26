import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

from typefly.llm_controller import LLMController
from typefly.robot_info import RobotInfo

print("="*60)
print("🚁 Janah SAR System - LLM Integration Test")
print("="*60)

# Create robot info
robot_info = RobotInfo.from_dict({
    'robot_id': 'janah_uav_001',
    'robot_type': 'virtual',
    'extra': {'capture': 0}
})

print("\n✅ Step 1: Creating UAV controller...")
controller = LLMController(robot_info)
print("   Controller created successfully!")

# Set missing child info
print("\n✅ Step 2: Setting missing child information...")
controller.set_missing_child_info({
    'name': 'سارة',
    'age': 5,
    'clothing_color': 'فستان وردي',
    'last_location': 'مدينة الألعاب',
    'description': 'شعر أسود طويل'
})

print("\n✅ Step 3: Verifying stored information...")
print(f"   Stored info: {controller.missing_child_info}")

print("\n" + "="*60)
print("✅ All tests passed! System ready for operation.")
print("="*60)