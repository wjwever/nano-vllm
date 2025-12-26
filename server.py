import asyncio
import websockets
import json
import torch
import multiprocessing as mp  # 修改点 1
from engine import WsEngine
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# ---------------------------------------------------------
# 模型推理进程 (Consumer)
# ---------------------------------------------------------
def inference_worker(input_queue, response_queues):
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = WsEngine(path, enforce_eager=False, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)
    print("Inference model loaded successfully in subprocess.")
    
    while True:
        # 获取任务
        while True:
            try:
                uid, txt = input_queue.get_nowait()
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": txt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                llm.add_request(prompt, sampling_params, uid)   
            except:
                break

        if llm.is_finished():
            time.sleep(0.01)
            continue

        seqs = llm.step()

        for seq in seqs:
            q = response_queues.get(seq.uid)
            if not q:
                continue
            inc_text = seq.inc_text(tokenizer)
            if inc_text:
                logging.info(f"inc_text:{inc_text}" )
                q.put({"type": "token", "content": inc_text})
            if seq.is_finished:
                q.put({"type": "done"})

# ---------------------------------------------------------
# WebSocket Server (Producer)
# ---------------------------------------------------------
class WSServer:
    def __init__(self, input_queue, response_queues):
        self.input_queue = input_queue
        self.response_queues = response_queues

    async def response_handler(self, websocket, user_id, q):
        """异步监听推理进程发回的消息"""
        loop = asyncio.get_running_loop()
        while True:
            # 队列 get 是阻塞的，必须在线程池运行
            msg = await loop.run_in_executor(None, q.get)
            await websocket.send(json.dumps(msg))
            if msg["type"] == "done":
                break

    async def ws_handler(self, websocket):
        user_id = id(websocket)
        # 必须使用 Manager 提供的 Queue 才能跨进程共享字典
        manager = mp.Manager()
        user_q = manager.Queue()
        self.response_queues[user_id] = user_q
        
        print(f"User {user_id} connected.")
        try:
            async for message in websocket:
                data = json.loads(message)
                self.input_queue.put((user_id, data["text"]))
                await self.response_handler(websocket, user_id, user_q)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.response_queues.pop(user_id, None)
            print(f"User {user_id} disconnected.")

async def main():
    # 设置 mp 使用 spawn 模式，解决 CUDA re-init 问题
    mp.set_start_method('spawn', force=True) # 修改点 2
    
    manager = mp.Manager()
    input_queue = manager.Queue()
    response_queues = manager.dict() # 使用 Manager 字典实现跨进程可见
    
    # 启动推理子进程
    worker_proc = mp.Process(target=inference_worker, args=(input_queue, response_queues))
    worker_proc.start()
    
    server = WSServer(input_queue, response_queues)
    
    print("Server starting on ws://localhost:8765")
    async with websockets.serve(server.ws_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main()) # 修改点 3
    except KeyboardInterrupt:
        print("Stopping server...")