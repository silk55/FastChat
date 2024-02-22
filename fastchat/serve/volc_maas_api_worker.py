"""
A model worker that calls volc mass api
Register models in a JSON file with the following format:
{
    "skylark-pro-public": {
        "model_path": "volc_maas",
        "host": "maas-api.ml-platform-cn-beijing.volces.com",
        "ak": "xxxx",
        "sk": "xxxx"
        "context_length": 2048,
        "region": "cn-beijing",
        "model_names": "skylark-pro-public",
    }
}
"model_path", "api_base", "token", and "context_length" are necessary, while others are optional.
"model_path" must contain volc_maas
"""
import argparse
import asyncio
import json
import uuid
from typing import List, Optional

import requests
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from volcengine.maas import MaasService, MaasException, ChatRole

from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.utils import build_logger
import os
from pprint import pprint

default_region = "cn-beijing"
default_host = "maas-api.ml-platform-cn-beijing.volces.com"


worker_id = str(uuid.uuid4())[:8]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

workers = []
worker_map = {}
app = FastAPI()


# reference to
def get_gen_kwargs(
    params,
    version: str = None,
    endpoint_id: str = None,
    seed: Optional[int] = None,
):
    stop = params.get("stop", None)
    if isinstance(stop, list):
        stop_sequences = stop
    elif isinstance(stop, str):
        stop_sequences = [stop]
    else:
        stop_sequences = []
    # prompt is not necessary, here we just using conv from base worker
    prompt = params["prompt"]

    top_k = params.get('top_k', 0)
    # maas need to ensure it >0 <100 
    top_k = max(0, min(top_k, 100))
    
    reqC = ModelRequest(model_name=params["model"],
                        version=version,
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        endpoint_id=endpoint_id,
                        max_new_tokens=params["max_new_tokens"],
                        top_k=top_k,
                        functions=params.get("functions", None)
                        )

    reqC.parse_prompt(prompt)
    # document: "https://www.volcengine.com/docs/82379/1099475"
    req = reqC.get_request()
    # no using
    gen_kwargs = {
        "do_sample": True,
        "return_full_text": bool(params.get("echo", False)),
        "max_new_tokens": int(params.get("max_new_tokens", 256)),
        "top_p": float(params.get("top_p", 1.0)),
        "temperature": float(params.get("temperature", 1.0)),
        "stop_sequences": stop_sequences,
        "repetition_penalty": float(params.get("repetition_penalty", 1.0)),
        "top_k": params.get("top_k", None),
        "seed": seed,
        "req": req,
        "functions": params.get("functions", None)
    }
    return gen_kwargs


def could_be_stop(text, stop):
    for s in stop:
        if any(text.endswith(s[:i]) for i in range(1, len(s) + 1)):
            return True
    return False


class ModelRequest:
    def __init__(self, model_name='skylark2-pro-4k',version=None,endpoint_id=None, max_new_tokens=15, temperature=0.5, top_p=0.9, top_k=0, functions=None):
        self.req = {
            "model": {
                "name": model_name
            },
            "parameters": {
                "max_new_tokens": max_new_tokens,  
                "temperature": temperature, 
                "top_p": top_p, 
                "top_k": top_k, 
            },
            "messages": [],
        }
        if version and len(version):
            self.req['model']['version'] = version
        if endpoint_id and len(endpoint_id):
            self.req["model"]["endpoint_id"] = endpoint_id
        if functions and len(functions):
            self.req['functions'] = functions
            

    def add_message(self, role, content, name = None, function_call = None):
        if isinstance(content, list):
            content = " ".join(map(str, content))
        self.req["messages"].append({
            "role": role,
            "content": content,
        })
        if name:
            self.req["messages"][-1]['name'] = name
        if function_call:
            self.req["messages"][-1]['function_call'] = function_call

    def append_last(self, content):
        if len(self.req["messages"]) == 0:
            return
        self.req["messages"][-1]["content"] += content

    def get_request(self):
        return self.req

    # hackable parser
    def parse_prompt(self, prompt):
        default_system = 'SYSTEM*+-'
        default_user = 'USER*+-'
        default_assistant = 'ASSISTANT*+-'
        try:
            parts = json.loads(prompt)
            for part in parts:
                if part['role'] == default_system:
                    self.add_message(ChatRole.SYSTEM, part['content'])
                if part['role'] == default_user:
                    self.add_message(ChatRole.USER, part['content'])
                if part['role'] == default_assistant:
                    self.add_message(ChatRole.ASSISTANT,part['content'],function_call=part.get("function_call", None))
                if part['role'] == 'function':
                    # for volc, when get function call we need also add assitant message before
                    if part.get('content',None) and part.get('name',None):
                        if len(self.req['messages']) > 0 and self.req['messages'][-1]['role'] != ChatRole.ASSISTANT:
                            self.add_message(ChatRole.ASSISTANT, "", function_call={
                                    "name": part['name'],
                                })
                        self.add_message(ChatRole.FUNCTION, part['content'], name=part['name'])
                        
        except json.JSONDecodeError as e:
            self.add_message(ChatRole.USER,prompt)
        # parse_failed, directly do
        if len(self.req["messages"]) == 0:
            self.add_message(ChatRole.USER,prompt)


class VolcMaasApiWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        host: str,
        ak: str,
        sk: str,
        region: str,
        context_length: int,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: Optional[str] = None,
        seed: Optional[int] = None,
        version: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        self.model_path = model_path
        self.host = host
        self.ak = ak
        self.sk = sk
        self.context_len = context_length
        self.seed = seed
        self.region = region
        self.version = version
        self.endpoint_id = endpoint_id

        logger.info(
            f"Connecting with volc api {self.model_path} as {self.model_names} on worker {worker_id} ..."
        )

        if not no_register:
            self.init_heart_beat()

    # todo
    def count_token(self, params):
        # No tokenizer here
        ret = {
            "count": 0,
            "error_code": 0,
        }
        return ret

    def generate_stream_gate(self, params):
        self.call_ct += 1

        gen_kwargs = get_gen_kwargs(params,version=self.version,endpoint_id=self.endpoint_id,seed=self.seed)
        stop = gen_kwargs["stop_sequences"]

        logger.info(f"request: {gen_kwargs['req']}")

        try:
            host = default_host if self.host == "" else self.host
            region = default_region if self.region == "" else self.region
            volc_ak = os.getenv("VOLC_ACCESSKEY") if self.ak == "" else self.ak
            volc_sk = os.getenv("VOLC_SECRETKEY") if self.sk == "" else self.sk
            client = MaasService(host=host, region=region)
            client.set_ak(volc_ak)
            client.set_sk(volc_sk)
            res = client.stream_chat(gen_kwargs["req"])

            reason = None
            function_call = None
            text = ""
            for chunk in res:
                if chunk.choice is not None and chunk.choice.message is not None:
                    if chunk.choice.message.content is not None:
                        text += chunk.choice.message.content
                    if not function_call and chunk.choice.message.function_call is not None:
                        # for skylark we get function call in single response while it is finish
                        function_call = chunk.choice.message.function_call
                        reason = "stop"
                s = next((x for x in stop if text.endswith(x)), None)
                if s is not None:
                    text = text[: -len(s)]
                    reason = "stop"
                    break
                if could_be_stop(text, stop):
                    continue
                if (
                    chunk.choice is not None and
                    chunk.choice.finish_reason is not None
                ):
                    reason = chunk.choice.finish_reason
                if reason not in ["stop", "length","function_call"]:
                    reason = None
                # the return of usage is not stable, so we may need a gpt tokenizer to gen it again 
                ret = {
                    "text": text,
                    "function_call": function_call,
                    "error_code": 0,
                    "finish_reason": reason,
                    "usage": chunk.usage,
                }
                yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())

    def get_embeddings(self, params):
        raise NotImplementedError()
    
    def get_rerank(self, params):
        raise NotImplementedError


def release_worker_semaphore(worker):
    worker.semaphore.release()


def acquire_worker_semaphore(worker):
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(worker):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(lambda: release_worker_semaphore(worker))
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    generator = worker.generate_stream_gate(params)
    background_tasks = create_background_tasks(worker)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    output = worker.generate_gate(params)
    release_worker_semaphore(worker)
    return JSONResponse(output)


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    await acquire_worker_semaphore(worker)
    embedding = worker.get_embeddings(params)
    release_worker_semaphore(worker)
    return JSONResponse(content=embedding)


@app.post("/worker_get_rerank")
async def api_get_rerank(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    worker = worker_map[params["model"]]
    rank = worker.get_rerank(params)
    release_worker_semaphore()
    return JSONResponse(content=rank)

@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return {
        "model_names": [m for w in workers for m in w.model_names],
        "speed": 1,
        "queue_length": sum([w.get_queue_length() for w in workers]),
    }


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    params = await request.json()
    worker = worker_map[params["model"]]
    return {"context_length": worker.context_len}


def create_volc_maas_api_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    # all model-related parameters are listed in --model-info-file
    parser.add_argument(
        "--model-info-file",
        type=str,
        required=True,
        help="maas model's info file path",
    )

    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    args = parser.parse_args()

    with open(args.model_info_file, "r", encoding="UTF-8") as f:
        model_info = json.load(f)

    logger.info(f"args: {args}")

    model_path_list = []
    host_list = []
    ak_list = []
    sk_list = []
    context_length_list = []
    model_names_list = []
    conv_template_list = []
    region_list = []
    version_list = []
    endpoint_id_list = []

    for m in model_info:
        model_path_list.append(model_info[m]["model_path"])
        host_list.append(model_info[m]["host"])
        ak_list.append(model_info[m]["ak"])
        sk_list.append(model_info[m]["sk"])
        t_region = model_info[m].get("region","")
        region_list.append(t_region)
        t_version = model_info[m].get("version","")
        version_list.append(t_version)
        t_endpoint = model_info[m].get("endpoint_id","")
        endpoint_id_list.append(t_endpoint)

        context_length = model_info[m]["context_length"]
        model_names = model_info[m].get("model_names", [m.split("/")[-1]])
        if isinstance(model_names, str):
            model_names = [model_names]
        conv_template = model_info[m].get("conv_template", None)

        context_length_list.append(context_length)
        model_names_list.append(model_names)
        conv_template_list.append(conv_template)

    logger.info(f"Model paths: {model_path_list}")
    logger.info(f"Context lengths: {context_length_list}")
    logger.info(f"Model names: {model_names_list}")
    logger.info(f"Conv templates: {conv_template_list}")

    for (
        model_names,
        conv_template,
        model_path,
        host,
        ak,
        sk,
        region,
        context_length,
        version,
        endpoint_id,
    ) in zip(
        model_names_list,
        conv_template_list,
        model_path_list,
        host_list,
        ak_list,
        sk_list,
        region_list,
        context_length_list,
        version_list,
        endpoint_id_list
    ):
        m = VolcMaasApiWorker(
            args.controller_address,
            args.worker_address,
            worker_id,
            model_path,
            host,
            ak,
            sk,
            region,
            context_length,
            model_names,
            args.limit_worker_concurrency,
            no_register=args.no_register,
            conv_template=conv_template,
            seed=args.seed,
            version=version,
            endpoint_id=endpoint_id,
        )
        workers.append(m)
        for name in model_names:
            worker_map[name] = m

    # register all the models
    url = args.controller_address + "/register_worker"
    data = {
        "worker_name": workers[0].worker_addr,
        "check_heart_beat": not args.no_register,
        "worker_status": {
            "model_names": [m for w in workers for m in w.model_names],
            "speed": 1,
            "queue_length": sum([w.get_queue_length() for w in workers]),
        },
    }
    r = requests.post(url, json=data)
    assert r.status_code == 200

    return args, workers


if __name__ == "__main__":
    args, workers = create_volc_maas_api_worker()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")