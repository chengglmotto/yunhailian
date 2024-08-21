#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/19 15:39 
@DESC     ：向量库
'''
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
class MyVector:

    def __init__(self,docs,embedding_model_path='BAAI/bge-small-zh-v1.5'):
        embedding = HuggingFaceEmbeddings(embedding_model_path)
        self.db = FAISS.from_texts(texts=docs,embedding=embedding)

    def search(self,text,top_k=100):
        result = [doc.page_content for doc in FAISS.similarity_search(query=text,top_k=top_k)]
        return result