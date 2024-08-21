#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/19 12:59 
@DESC     ：工具类,提供了解析excel文档的函数load_excel
'''
import numpy as np
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter

def load_excel_1(df:pd.DataFrame):
    df_1 = df['狗狗介绍']
    df_1_head_list = df_1.columns.tolist()
    df_1_head_list = df_1_head_list[:-1]
    print(df_1_head_list)
    lines = []
    for idx,line in df_1.iterrows():
        desc = ''
        for head in df_1_head_list:
            if type(line[head]) is str:
                desc+= head +' '+line[head]
        desc = "".join(desc.split())
        lines.append(desc)

    return lines

def load_excel_2(df:pd.DataFrame):
    df_2 = df['更了解狗狗']
    df_2_head_list = df_2.columns.tolist()
    lines = []
    for idx, line in df_2.iterrows():
        question = line['问题']
        answer = line['解答']
        reason = line['原因']
        if type(answer) is not str or type(question) is not str:
            continue
        if type(reason) is not str:
            desc = "问题:"+question + ',答案:' + answer
        else:
            desc = "问题:"+question + ',答案:' + answer+',原因:'+reason
        lines.append(desc)

    return lines

def load_excel_3(df:pd.DataFrame):
    df_3 = df['通用语']
    lines = []
    for idx,line in df_3.iterrows():
        class_ = line['分类']
        message_1 = line['设置话术1']
        message_2 = line['回复话术2']
        message_3 = line['回复话术3']
        message_4 = line['回复话术4']
        if type(message_1) is str:
            lines.append(message_1+",这段话的类别是:"+class_)
        if type(message_2) is str:
            lines.append(message_2+",这段话的类别是:"+class_)
        if type(message_3) is str:
            lines.append(message_3 + ",这段话的类别是:" + class_)
        if type(message_4) is str:
            lines.append(message_4 + ",这段话的类别是:" + class_)

    return lines

def load_excel_4(df:pd.DataFrame):
    df_4 = df['评价回复']
    lines = []
    for idx, line in df_4.iterrows():
        class_ = line[0]
        desc = line[1]
        for i in range(2,5):
            message = line[i]
            if type(message) is str and type(class_) is str and type(desc) is str:
                lines.append(message + ";这段话的产品类型是:" + class_+",评价类型是:"+desc)
            elif type(message) is str and type(class_) is str:
                lines.append(message + ";这段话的产品类型是:" + class_)
            else:
                lines.append(message)

    return lines

def load_excel_5(df:pd.DataFrame):
    df_5 = df['物流售后']
    lines = []
    for idx, line in df_5.iterrows():
        lines.append('问题:'+line[2]+',其回复话术/方式:'+line[3])
    return lines

def load_excel_6(df:pd.DataFrame):
    df_6 = df['产品清单']
    lines = []
    pre_question = ''
    pre_class = ''
    for idx, line in df_6.iterrows():
        if idx < 8 or idx > 52:
            continue
        class_ = line[1]
        question = line[2]
        content = line[3]
        if type(question) is str:
            line = '回复话术:'+content+',对应问题是:'+question
            pre_question = question
            if type(class_) is str:
                line +=',对应类别:'+class_
                pre_class = class_
            else:
                line+=',对应类别:'+pre_class
        else:
            line = '回复话术:'+content+',对应问题是:'+pre_question+',对应类别:'+pre_class
        lines.append(line)
        if idx >55:
            class_1 = line[0]
            name = line[1]
            materail =  line[2]
            desc = line[3]
            # print(class_1)
            # print(name)
            # print(materail)
            # print(desc)
            if type(class_1) is not str:
                class_1 = '宠物用品'
            line = '产品:'+name+',类别:'+class_1+',材质:'+materail+',卖点:'+desc
            lines.append(line)

    return lines
def load_excel_7(df:pd.DataFrame):
    df_7 = df['狗狗玩具售后']
    lines = []
    for idx,line in df_7.iterrows():
        lines.append('处理方法:'+line[1]+',售后问题:'+line[0])

    return lines
#df = pd.read_excel('F:\PycharmProjects\yunhailian\dataset\客服知识库.xlsx', sheet_name=None)


def load_excel(file_path):
    df = pd.read_excel(file_path, sheet_name=None)
    lines = load_excel_1(df)
    lines.extend(load_excel_2(df))
    lines.extend(load_excel_3(df))
    lines.extend(load_excel_4(df))
    lines.extend(load_excel_5(df))
    lines.extend(load_excel_6(df))
    lines.extend(load_excel_7(df))
    lines = [l+'</>' for l in lines if type(l) is str]
    print('加载excel文档，获得{.d}条知识',len(lines))
    return lines


if __name__ == '__main__':
    lines = load_excel('F:\PycharmProjects\yunhailian\documents\客服知识库.xlsx')
    f = open('log.txt', 'w',encoding='utf-8')
    for i in range(len(lines)):
        if type(lines[i]) is not str:
            continue
        f.write(lines[i] + '</>\n')
    f.close()



