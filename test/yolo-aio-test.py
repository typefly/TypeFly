import aiohttp
import json, os
import asyncio

EDGE_SERVICE_IP = os.environ.get("EDGE_SERVICE_IP", "localhost")

async def detect():
    image_path = '../test/images/kitchen.webp'
    
    config = {
        'robot_info': '{"robot_id": "robot1", "robot_type": "drone", "ip": "127.0.0.1"}',
        'service_type': 'yolo',
        'tracking_mode': False,
        'image_id': 1,
        'conf': 0.3
    }

    form_data = aiohttp.FormData()
    form_data.add_field('image', open(image_path, 'rb'), filename='kitchen.webp', content_type='image/webp')
    form_data.add_field('json_data', json.dumps(config), content_type='application/json')

    async with aiohttp.ClientSession() as session:
        print("Sending request")
        async with session.post(f"http://{EDGE_SERVICE_IP}:50049/process", data=form_data) as response:
            content = await response.text()
            print("Received response")
            print(content)

if __name__ == "__main__":
    asyncio.run(detect())  # If using Jupyter, replace with `await detect()`
