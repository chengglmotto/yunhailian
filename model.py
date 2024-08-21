#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/19 11:11 
@DESC     ：模型类
'''
from modelscope import AutoTokenizer,AutoModelForCausalLM
import torch

device= 'cuda' if torch.cuda.is_available() else 'cpu'
class QwenLLM():

    def __init__(self,model_path=''):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype='auto',
            device_map = 'auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

    def chat(self,input_text):
        model_inputs = self.tokenizer([input_text],return_tensors='pt').to(device)
        generate_ids = self.model.generate(model_inputs.input_ids,max_new_tokens=400,repetition_penalty=1.25)

        generate_ids = [
            output_ids[len(input_ids)] for input_ids,output_ids in zip(model_inputs.input_ids,generate_ids)
        ]

        return self.tokenizer.batch_decode(generate_ids,skip_special_tokens=True)[0]



