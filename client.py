import asyncio
import websockets
import json

async def translate():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            text = input("\n请输入(q 退出): ")
            if text.lower() == 'q':
                break
                
            # 发送请求
            await websocket.send(json.dumps({"text": text}))
            
            print("对话结果: ", end="", flush=True)
            
            # 循环接收流式词汇
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if data["type"] == "token":
                    print(data["content"], end="", flush=True)
                elif data["type"] == "done":
                    break

if __name__ == "__main__":
    try:
        asyncio.run(translate())
    except KeyboardInterrupt:
        pass
