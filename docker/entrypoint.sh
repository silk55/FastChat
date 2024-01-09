#!/bin/bash

# Ensure log directory exists
mkdir -p /var/log/fastchat

if [[ "${C_MODE}" == "manager" ]]; then
supervisord -c /supervisor/fastchat-manager.conf
supervisord -n -c /supervisor/fastchat-openai.conf
elif [[ "${C_MODE}" == "volc" ]]; then
supervisord -n -c /supervisor/fastchat-maas-worker.conf
else
supervisord -n -c /supervisor/fastchat-llm-worker.conf
fi
