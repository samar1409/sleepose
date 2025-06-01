import asyncio
import websockets
import json
from datetime import datetime

class AppleWatchConnector:
    def __init__(self):
        self.latest_data = {}
        self.connected = False
        
    async def handle_client(self, websocket, path):
        self.connected = True
        print("Apple Watch connected!")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                self.latest_data = {
                    'timestamp': datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                    'heart_rate': data['heart_rate'],
                    'source': 'apple_watch_real'
                }
                print(f"Received data: HR={data['heart_rate']}")
        except websockets.ConnectionClosed:
            self.connected = False
            print("Apple Watch disconnected")
            
    def get_latest_data(self):
        return self.latest_data
    
    def is_connected(self):
        return self.connected

    async def start_server(self):
        async with websockets.serve(self.handle_client, "localhost", 8501):
            await asyncio.Future()  # run forever

watch_connector = AppleWatchConnector()

if __name__ == "__main__":
    asyncio.run(watch_connector.start_server())