import aiohttp
import json
import asyncio
async def detect():
    image = open('../test/images/kitchen.webp', 'rb')
    config = {
        'robot_info': '{"robot_id": "robot", "robot_type": "drone"}',
        'service_type': 'yolo',
        'tracking_mode': False,
        'conf': 0.3
    }
    files = {
        'image': image,
        'json_data': json.dumps(config)
    }
    print(files)
    async with aiohttp.ClientSession() as session:
        print("Sending request")
        async with session.post("http://0.0.0.0:50049/process", data=files) as response:
            content = await response.text()
            print("Received response")
            print(content)

if __name__ == "__main__":
    asyncio.run(detect())