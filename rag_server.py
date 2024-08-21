#!/usr/bin/env pyathon
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/19 11:11 
@DESC     ：rag服务端,提供了一个进行query的http接口,启动此服务的时候，就已经把文件建好了
'''
from embedding_server import EmbeddingServer
from flask import Flask,request,session
import logging
import socket
from argparse import ArgumentParser

app = Flask(__name__)

@app.route("/rag",methods=['POST'])
def client():
    data = request.json.get("message")
    app.logger.info("request", data)
    response = db.query(data)
    app.logger.info("response", response)
    return response

if __name__ == '__main__':
    parser = ArgumentParser("rag_server")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    try:
        file_path = 'documents/客服知识库.xlsx'
        db = EmbeddingServer([file_path],'BAAI/bge-reranker-large')
        ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        print(ip)
        app.run(host=ip, port=args.port, debug=True)
    except Exception as e:
        print(e)
