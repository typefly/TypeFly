import grpc
import asyncio
import time

class ServiceManager:
    def __init__(self):
        self.services = {}
        self.channel_queues = {}
        self.channels_initialized = False
        self.dedicated_channels = {}
        self.dedicated_channels_timeout = 10
        self.last_cleanup = time.time()

    def add_service(self, service_name, host, ports):
        self.services[service_name] = (host, ports.split(","))
        self.channel_queues[service_name] = asyncio.Queue()

    async def _initialize_channels(self):
        if self.channels_initialized:
            return
        for service_name, (host, ports) in self.services.items():
            queue = self.channel_queues[service_name]
            for port in ports:
                channel = grpc.aio.insecure_channel(f"{host}:{port}")
                await queue.put(channel)
        self.channels_initialized = True

    async def clean_dedicated_channels(self):
        if time.time() - self.last_cleanup > self.dedicated_channels_timeout:
            self.last_cleanup = time.time()
            
            users_to_remove = []
            services_to_remove = []

            # Collect items to remove
            for user_name, service_channels in self.dedicated_channels.items():
                print(f"Checking dedicated channels for user {user_name}")
                for service_name, (channel, timestamp) in service_channels.items():
                    if time.time() - timestamp > self.dedicated_channels_timeout:
                        print(f"Removing expired dedicated channel for user {user_name} and service {service_name}")
                        await self.channel_queues[service_name].put(channel)
                        services_to_remove.append((user_name, service_name))

                if len(service_channels) == len([service for user, service in services_to_remove if user == user_name]):
                    users_to_remove.append(user_name)

            # Perform deletions
            for user, service in services_to_remove:
                del self.dedicated_channels[user][service]
                if len(self.dedicated_channels[user]) == 0:
                    del self.dedicated_channels[user]

            for user in users_to_remove:
                if user in self.dedicated_channels:
                    del self.dedicated_channels[user]

    async def get_service_channel(self, service_name, dedicated=False, user_name=None):
        await self._initialize_channels()  # Ensure channels are initialized

        await self.clean_dedicated_channels()
        
        if dedicated:
            # If there is no dedicated channel for this user, get one from the queue
            if user_name not in self.dedicated_channels:
                # If there is no more channel in the queue, return None
                if self.channel_queues[service_name].qsize() > 0:
                    self.dedicated_channels[user_name] = {service_name: (self.channel_queues[service_name].get_nowait(), time.time())}
                else:
                    return None
            # If there is a dedicated channel for this user, update time and return it
            else:
                self.dedicated_channels[user_name][service_name] = (self.dedicated_channels[user_name][service_name][0], time.time())
                return self.dedicated_channels[user_name][service_name][0]
        else:
            channel = await self.channel_queues[service_name].get()
            return channel

    async def release_service_channel(self, service_name, channel):
        await self.channel_queues[service_name].put(channel)