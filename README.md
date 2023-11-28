## intro 
fork from https://github.com/lm-sys/FastChat
and add support for volc maas, click https://www.volcengine.com/docs/82379/1099475 to see detail

## how to use

## how to install
* cd path
* pip install -e .

or install after build whl
* python -m build
* pip install fschat*.whl

### origin readme
original readme to see origin-readme.md

### all in one bash
see detailed serve/launch_all_serve.py


### cli to chat with a model

* GPU
  * python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2 --max-gpu-memory 8GiB

* CPU 
  * python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device cpu


### common method to run like openai interface format

* controller 
  * python -m fastchat.serve.controller --host 0.0.0.0 --port 21001
* openai server
  * python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
* worker （example）
  * like
  * CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path /data/chatglm2-6b  --host 127.0.0.1 --port 21006 --worker-address http://127.0.0.1:21006 
  * or like
  * python3 -m fastchat.serve.model_worker --model-path ~/data/Mistral-7B-Instruct-v0.1  --num-gpus 2 --max-gpu-memory 18Gib --host 127.0.0.1 --port 21002 --worker-address http://127.0.0.1:21002

* maas_worker
  
  * python -m fastchat.serve.volc_maas_api_worker --model-info-file "$MAAS_PATH" --worker-address http://127.0.0.1:21010 --controller-address http://127.0.0.1:21001 --host 0.0.0.0 --port 21010

  * $MAAS_PATH should be a json like:
```json
{
    "skylark-pro": {
        "model_path": "volc_maas",
        "host": "maas-api.ml-platform-cn-beijing.volces.com",
        "ak": "",
        "sk": "",
        "context_length": 2048,
        "model_names": "skylark-pro-public",
        "conv_template": null
    }
}
```

you can add several models to a single json.