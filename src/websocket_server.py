import asyncio
import websockets
import json
from datetime import datetime

class WatchDataServer:
    def __init__(self):
        self.latest_data = None
        self.clients = set()
        
    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def broadcast(self, data):
        if not self.clients:
            return
        await asyncio.gather(
            *[client.send(json.dumps(data)) for client in self.clients]
        )
    
    async def handler(self, websocket):
        """Handle incoming WebSocket connections"""
        await self.register(websocket)

async def main():
    server = WatchDataServer()
    async with websockets.serve(server.handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())