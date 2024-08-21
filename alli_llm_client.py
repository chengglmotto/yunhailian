#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/20 10:06 
@DESC     ：远程调用Qwen大模型，需要自己填写api_key
'''
from openai import OpenAI
import os

def get_response(message):
    client = OpenAI(
        api_key="", # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': message}],
        temperature=0.8,
        top_p=0.8
        )
    print(completion.model_dump_json())
    return completion

if __name__ == '__main__':
    get_response()