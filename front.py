#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/20 8:33 
@DESC     ：客服机器人前端页面,把
'''
import streamlit as st
import pandas as pd
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import requests
import alli_llm_client
import socket

@st.cache_resource
def get_chain():
    llm = HuggingFacePipeline.from_model_id(
        model_id="THUDM/chatglm3-6b",
        task="text-generation",
        device=0,
        model_kwargs={"trust_remote_code": True,
                      #   "temperature":0.9,
                      #   "do_sample":True
                      },
        pipeline_kwargs={"max_new_tokens": 5000},
    )
    template = """{question}"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    return chain

def greet2(name):
    response = get_chain().invoke({"question": name})
    return response


# st.title('🦜🔗 Quickstart App')
st.set_page_config(page_title="Andy Chatbot")
with st.sidebar:
    st.title('Andy Chatbot')
    st.success('API key already provided!', icon='✅')
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    st.warning('Please enter your credentials!', icon='⚠️')
    st.success('Proceed to entering your prompt message!', icon='👉')
    st.subheader('Models and parameters')
    # selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B', 'Llama2-70B'],
    #                                       key='selected_model')
    # if selected_model == 'Llama2-7B':
    #     llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    # elif selected_model == 'Llama2-13B':
    #     llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    # else:
    #     llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    st.markdown(
        '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "亲亲您好，我在的哦，请问有什么可以帮您的呢?"}]

    # Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def rag_client(message):
    ip = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    url = "http://" + str(ip) + ":8092/rag"
    headers = {'Content-Type': 'application/json'}
    params = {"message",message}
    response =  requests.post(url,data=params,headers=headers)
    return response

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    prompt_class = "你是一名电商客服人员,请分辨某句话是否属于电商客服领域的问题，如果是返回是,如果不是返回否，例如: <example>提问-请问我网购的东西到哪里了? '是'</example><example>提问-一加一等于几 '否'<example/>,回答只返回1个字,需要回复句子如下:{0}"
    prompt_cautious = "你是一名电商客服人员,现在有上下文信息:1.{0},2.{1},3.{2},请参照上文上下文信息,对下面的话进行回答。请记住:\n--答案中不能包含'最大'、'唯一'等违反广告法的宣传语;\n--如果问题包含退款，则需要提示转人工。这段话进行回答:{4}"
    prompt_query = "你是一名电商客服人员,现在有上下文信息:1.{0},2.{1},3.{2},请参照这些上下文信息,对下面这段话进行回答:{4}"
    response = test_llm_ali.get_response(prompt_class.format(prompt_input))
    print(response)
    if response.choices[0].message.content == '是':
        rag_resp_list = rag_client(prompt_input)
        if len(rag_resp_list) == 0:
            response = test_llm_ali.get_response(prompt_cautious.format(rag_resp_list[0],rag_resp_list[1],rag_resp_list[2],prompt_cautious.format(prompt_input)))
        else:
            response = test_llm_ali.get_response(prompt_query.format(rag_resp_list[0],rag_resp_list[1],rag_resp_list[2],prompt_input))
    else:
        return "不好意思，亲,本机器人目前只能回答电商客服领域问题，还请检查"
    return response.choices[0].message.content

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

#调用大模型 判断是否客服
#调用大模型,加载特殊prompt
#调用大模型之前,进行RAG Query,没有命中或者包含退货、退款,返回“您好,接下来转接人工服务”

