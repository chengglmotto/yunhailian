#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Author  ：chenggl
@Date    ：2024/8/21 15:08 
@DESC     ：评测函数，提供了最小编辑距离、Jaccard相似度和rouge三个评测方法
'''
from rouge import Rouge
import jieba



def minEditDistance(word1, word2):
	m = len(word1) + 1
	n = len(word2) + 1
	dp = [[0 for i in range(n)] for j in range(m)]

	for i in range(n):
		dp[0][i] = i
	for i in range(m):
		dp[i][0] = i
	for i in range(1, m):
		for j in range(1, n):
			if word1[i-1] == word2[j-1]:
				dp[i][j] = dp[i-1][j-1]
			else:
				dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1, dp[i-1][j-1]+2)

	return dp[m-1][n-1]


def jaccard_similarity(list1, list2):
	intersection = len(set(list1).intersection(list2))
	union = len(set(list1)) + len(set(list2)) - intersection
	return intersection / float(union)


# word1 = generate_text
# print(word1)
# word2 = groud_truth_test
# print(word2)
# print(jaccard_similarity(word1, word2))

def rouge_eval(generate_text,groud_truth_test):

	rouge = Rouge()
	scores = rouge.get_scores(' '.join(jieba.lcut(generate_text)), ' '.join(jieba.lcut(groud_truth_test)))
	print(scores)
	# 打印结果
	print("ROUGE-1 precision:", scores[0]["rouge-1"]["p"])
	print("ROUGE-1 recall:", scores[0]["rouge-1"]["r"])
	print("ROUGE-1 F1 score:", scores[0]["rouge-1"]["f"])

import pandas as pd

df = pd.read_excel('documents/评测数据集.xlsx')
for idx,item in df.iterrows():
	rouge_eval(item[1],item[2])