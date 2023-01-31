import os
import emoji
import re
import pandas as pd
from tqdm import tqdm
import jieba
import wordcloud # 词云展示库
from wordcloud import *
import csv
import numpy as np
from PIL import Image # 图像处理库
import matplotlib.pyplot as plt # 图像展示库
import collections # 词频统计库
import json
from sklearn.feature_extraction.text import CountVectorizer
import jieba.posseg
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
from slda import lda_gibbs_sampling1
warnings.filterwarnings("ignore")




with open('data/stop_words.txt','r',encoding='utf-8')as f:
    stopwords=f.readlines()
    stopwords=[x.strip() for x in stopwords]
# print(stopwords)
pos_dict = {}

def get_words(sen,train):
    #print(sen)
    new_words=[]
    sen=sen.replace(' ','').replace('/n','').replace('/t','').replace('\t','').replace('\n','')
    sen = emoji.demojize(sen)#去除emoji表情
    sen = re.sub(':\S+?:', '', sen)
    sen=sen.replace(' ','')

    results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
    sen = re.sub(results, '', sen)

    #cutwords1 = jieba.lcut(sen)  # 分词
    cutwords1 = jieba.posseg.lcut(sen)  # 分词
    for x in cutwords1:
        pos_dict[x.word]=x.flag
    cutwords1=[x.word for x in cutwords1]

    #print(cutwords1)
    interpunctuations = [',','.','’','…','-',':','~', ';', '"','?',"'s", '(', ')','...', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义符号列表
    cutwords2 = [word for word in cutwords1 if word not in interpunctuations]  # 去除标点符号
    stops = set(stopwords)#停用词
    stops=list(stops)
    stops=stops+["n't","''",'rt','1',' ','评价','"','｀','～','\n','/','ω',')','(','_','＝','=','?','??','I','|']#可以在stop_words.txt里加入停用词，也可以在这里写入
    cutwords3 = [word for word in cutwords2 if word not in stops]
    cutwords3=[x for x in cutwords3 if x.isdigit() is False]

    jcutwords3 = ' '.join(cutwords3).strip()
    if train==1:
        return jcutwords3

    else:
        return cutwords3



def get_vob(df):
    # vectorizer = CountVectorizer(decode_error="replace")  # 最大特征词数
    # vec_train = vectorizer.fit_transform(texts)  # 转换好的词频矩阵
    # feature_path = 'feature.pkl'
    # with open(feature_path, 'wb') as fw:
    #     pickle.dump(vectorizer.vocabulary_, fw)

    texts=df['item'].to_list()
    alltext=[]

    n=0
    vob_dict={}
    vob_dict['sep_']=0
    for x in tqdm(texts):
        for y in x.split('。'):
            #print(y)
            # if len(y)>=1:
            #     alltext.append(get_words(y,1))
            for z in get_words(y,0):
                if z not in vob_dict.keys():
                    vob_dict[z]=n
                    n=n+1

    # print(alltext)
    # vec_train = vectorizer.fit_transform(alltext)
    # vob_dict=vectorizer.vocabulary_

    return vob_dict

def text_index(text,vob_dict):
    res=[]
    for x in text:
        if x in vob_dict.keys():
            res.append(vob_dict[x])
        else:
            res.append(vob_dict['sep_'])
    #print(np.array(res,dtype=np.int32))
    return np.array(res,dtype=np.int32)


def docto_index(df,vob_dict):
    docs=df['item'].to_list()
    alldocinde=[]
    for doc in tqdm(docs,'文档转换索引...'):
        doccut=[get_words(x,0) for x in doc.split('。')]
        #print(doccut)
        doccut=[x for x in doccut if len(x)>=1]
        # print(doccut)
        #print('*********************')
        docindex = [text_index(x,vob_dict) for x in doccut]
        docindex=np.array(docindex,dtype=object)
        #print(docindex)
        alldocinde.append(docindex)
        #print(docindex)
        #print(docindex.shape)
        # for s in doc.split(','):
        #     print(s)
    alldocinde=np.array(alldocinde,dtype=object)
    #print(alldocinde)
    return alldocinde


df=pd.read_excel('./data/1.10数据.xlsx').iloc[:100,:]
print(df)
vob_dict=get_vob(df)

print('词表长度:',len(vob_dict.keys()))
ldadata=docto_index(df,vob_dict)

topics =25#主题数量，自己调整来跑
alpha, beta = 0.5 / float(topics), 0.5 / float(topics)
iterations = 5#训练次数，自己调整
lda = lda_gibbs_sampling1(K=topics, alpha=alpha, beta=beta, docs=ldadata, V=len(vob_dict.keys()))

for i in range(iterations):
    lda.inference()
    if i % 10 == 0:
        print("Iteration:", i, "Perplexity:", lda.perplexity())
        # features = lda.heldOutPerplexity(ldadata, 3)
        # print ("Held-out:", features[0])
        res=lda.topicdist()
        maxres=[np.argmax(x) for x in res]

        res=pd.DataFrame(res)
        res.columns=['Topic_'+str(x) for x in range(topics)]
        res['max_Topic']=maxres

        resdf=pd.concat([df,res], axis=1)
        resdf.to_csv('result.csv',index=None)
