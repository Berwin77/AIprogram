#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import math 
import random 
from pandas import DataFrame


# In[2]:


class UserBasedCF:
    def __init__(self, path):
        self.train= {}
        self.test = {}
        self.generate_dataset(path)
    def loadfile(self, path):
        with open(path, "r", encoding='utf-8') as fp:
            for i,line in enumerate(fp):
                yield line.strip("\r\n")
    def generate_dataset(self, path, pivot=0.7):
        # 读取文件，并声称用户-物品的评分表和测试表
        i = 0
        for line in self.loadfile(path):
            user,movie,rating,_ = line.split("::")
            if i<=10:
                print("{},{},{},{}".format(user,movie, rating,_))
            i += 1
            if random.random() < pivot:
                # 如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值。
                self.train.setdefault(user, {})
                self.train[user][movie] = int(rating)
            else:
                self.test.setdefault(user, {})
                self.test[user][movie] = int(rating) 
    
    def UserSimilarity(self):
        # 建立物品-用户的倒排表
        self.item_users = dict()
        for user,items in self.train.items():
            for i in items.keys():
                if i not in self.item_users:
                    self.item_users[i] = set()
                self.item_users[i].add(user)
        # 计算用户-用户共线矩阵
        C = dict() # 用户-用户共线矩阵
        N = dict() # 用户产生行为的物品个数
        for i,users in self.item_users.items():
            for u in users:
                N.setdefault(u,0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    if u==v:
                        continue
                    C[u].setdefault(v,0)
                    C[u][v] += 1 
        # 计算用户-用户相似度，这里用余弦相似度
        self.W = dict()
        for u,related_users in C.items():
            self.W.setdefault(u,{})
            for v, cuv in related_users.items():
                self.W[u][v] = cuv /math.sqrt(N[u]*N[v])
        return self.W, C, N
    
#     # 惩罚通用物品 如《新华字典》的影响
#     def UserSimilarity(self):
#     # 建立物品-用户的倒排表
#     self.item_users = dict()
#     for user,items in self.train.items():
#         for i in items.keys():
#             if i not in self.item_users:
#                 self.item_users[i] = set()
#             self.item_users[i].add(user)
#     # 计算用户-用户共线矩阵
#     C = dict() # 用户-用户共线矩阵
#     N = dict() # 用户产生行为的物品个数
#     for i,users in self.item_users.items():
#         for u in users:
#             N.setdefault(u,0)
#             N[u] += 1
#             C.setdefault(u, {})
#             for v in users:
#                 if u==v:
#                     continue
#                 C[u].setdefault(v,0)
#                 C[u][v] += (1/math.log(1+len(self.item_users(i)))
#     # 计算用户-用户相似度，这里用余弦相似度
#     self.W = dict()
#     for u,related_users in C.items():
#         self.W.setdefault(u,{})
#         for v, cuv in related_users.items():
#             self.W[u][v] = cuv /math.sqrt(N[u]*N[v])
#     return self.W, C, N
    
    
    # 给user推荐，前K个相关用户
    
    def Recommend(self, u, K=3, N=10):
        rank = dict()
        action_item = self.train[u].keys() #用户user产生过行为的item
        # v: 用户v
        #wuv：用户u和用户v的相似度
        for v, wuv in sorted(self.W[u].items(), key=lambda x:x[1], reverse=True)[:K]:
            ## 遍历前K个与user最相关的用户
            ## i：用户v有过行为的物品i
            ## rvi:用户v对物品i的打分
            for i,rvi in self.train[v].items():
                if i in action_item:
                    continue
                rank.setdefault(i,0)
                ## 用户对物品的感兴趣程度：用户u和用户v的相似度*用户v对物品i的打分
                rank[i] += wuv*rvi
        return dict(sorted(rank.items(),key=lambda x:x[1], reverse=True)[:N])
    
    
    def recallAndPrecision(self, k=8, nitem=10):
        hit = 0
        recall = 0
        precision = 0
        for user, items in self.test.items():
            rank = self.Recommend(user,K=k, N=nitem)
            hit += len(set(rank.keys())&set(items.keys()))
            recall += len(items)
            precision += nitem
        return (hit/(recall*1.0), hit/(precision*1.0))
                    


         


# In[7]:


# 将dict转换为Dataframe
def trans_dic_2_matrix(dic):
     return DataFrame(dic).T.fillna(0)

if __name__ == '__main__':
    # user, movie, rating, _
    path = '../ml-1m/ratings.dat'
    ucf = UserBasedCF(path)

    W,C,N = ucf.UserSimilarity()

    # 数据总用户数
    s = list(C.keys())

    print(len(s))
    print("########################")
    df_c = trans_dic_2_matrix(C)  
    # 查看 推荐 '520'用户 的前10个商品
    recomend = ucf.Recommend('520')
    print(recomend)
    # 计算 精确度 和 召回率
    ucf.recallAndPrecision()


# In[ ]:




