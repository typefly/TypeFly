import aiohttp
import json
import asyncio
async def detect():
    image = open('../test/images/kitchen.webp', 'rb')
    files = {
        'image': image,
        'json_data': json.dumps({'service': 'yolo'})
    }
    print(files)
    async with aiohttp.ClientSession() as session:
        print("Sending request")
        async with session.post("http://0.0.0.0:50048/yolo", data=files) as response:
            content = await response.text()
            print("Received response")
            print(content)

if __name__ == "__main__":
    asyncio.run(detect())