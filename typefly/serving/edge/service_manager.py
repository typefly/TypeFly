import grpc
import os, sys
import asyncio
import time, json

PROJ_DIR = os.environ.get("PROJ_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, PROJ_DIR)
from typefly.robot_info import RobotInfo

class ServiceManager:
    def __init__(self):
        self.service_info: dict[str, tuple[str, list[int]]] = {}
        self.service_channels: dict[str, asyncio.Queue] = {}
        self.channels_initialized: bool = False
        self.assigned_channels: dict[RobotInfo, dict[str, grpc.Channel]] = {}
        self.assigned_channels_timeout: int = 10
        self.last_cleanup: float = time.time()
        self.lock = asyncio.Lock()

    def add_service(self, service_type: str, host: str, ports: list[int]):
        self.service_info[service_type] = (host, ports)
        self.service_channels[service_type] = asyncio.Queue()

    async def _initialize_channels(self):
        if self.channels_initialized:
            return
        for service_type, (host, ports) in self.service_info.items():
            queue = self.service_channels[service_type]
            for port in ports:
                channel = grpc.aio.insecure_channel(f"{host}:{port}")
                await queue.put(channel)
        self.channels_initialized = True

    async def clean_dedicated_channels(self):
        if time.time() - self.last_cleanup > self.assigned_channels_timeout:
            self.last_cleanup = time.time()
            
            users_to_remove = []
            for robot_info, channels in self.assigned_channels.items():
                services_to_remove = []
                for service_type, channel_info in channels.items():
                    if time.time() - channel_info["timestamp"] > self.assigned_channels_timeout:
                        services_to_remove.append(service_type)
                        await self.service_channels[service_type].put(channel_info["channel"])

                for service_type in services_to_remove:
                    del channels[service_type]
                
                if not channels:
                    users_to_remove.append(robot_info)
            
            for robot_info in users_to_remove:
                del self.assigned_channels[robot_info]

    async def get_service_channel(self, service_type: str, robot_info: RobotInfo) -> str | grpc.Channel:
        async with self.lock:
            await self._initialize_channels()  # Ensure channels are initialized
            await self.clean_dedicated_channels()

            if service_type not in self.service_info:
                return "Service not found"

            if robot_info not in self.assigned_channels:
                self.assigned_channels[robot_info] = {}
            
            if service_type not in self.assigned_channels[robot_info]:
                channel_info = {}
                # check available channels for the given service
                try:
                    channel_info["channel"] = self.service_channels[service_type].get_nowait()
                    channel_info["timestamp"] = time.time()
                    self.assigned_channels[robot_info][service_type] = channel_info
                    return channel_info["channel"]
                except asyncio.QueueEmpty:
                    return f"No available channels for service {service_type}"
            else:
                channel_info = self.assigned_channels[robot_info][service_type]
                channel_info["timestamp"] = time.time()
                return channel_info["channel"]