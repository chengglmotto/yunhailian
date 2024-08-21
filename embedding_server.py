#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/19 15:37 
@DESC     ：embedding服务端,内置了embedding的初始化和若干操作
'''
from langchain.text_splitter import  CharacterTextSplitter

import utils
from vector import MyVector
from FlagEmbedding import FlagReranker
import torch
class EmbeddingServer:

    def __init__(self,file_paths,reranker_model_path):
        self.vs = self.save_embedding(file_paths[0])
        self.reranker = FlagReranker(model_name_or_path=reranker_model_path,use_fp16=True)


    def save_embedding(self,file_path):
        docs = self.load_excel_file(file_path)
        chunks = self.text_split(docs)
        self.vs = MyVector(chunks)

    def text_split(self,docs,chunk_size=500,chunk_over_lap=20):
        text_spliter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_over_lap)
        #text_spliter.split_text()
        chunks = []
        for idx,doc in enumerate(docs):
            print('------------------')
            if type(doc) is str:
                chunks.extend(text_spliter.split_text(doc))

        return chunks

    def query(self,query_text,top_k=100,score_threshold=1.5):
        candidates = self.vs.search(query_text,top_k=top_k)
        pairs = [(query_text, doc) for doc in candidates]
        scores = self.reranker.compute_score(pairs)
        return [scores[idx] for idx in  torch.topk(torch.tensor(scores),3).indices.data]

    def load_excel_file(self,file_path):
        docs = utils.load_excel(file_path)
        return docs