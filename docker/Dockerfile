FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04


COPY fschat-0.2.35-py3-none-any.whl /fschat-0.2.35-py3-none-any.whl
COPY supervisor/get-pip.py /get-pip.py

RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl supervisor 
RUN python3.9 /get-pip.py
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install fschat-0.2.35-py3-none-any.whl
RUN pip3 install fschat[model_worker,webui] pydantic==1.10.13
RUN pip3 install sentence-transformers

COPY model/  /model/
COPY supervisor/ /supervisor/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]