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
import pandas as pd
import jieba.posseg
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")




with open('data/stop_words.txt','r',encoding='utf-8')as f:
    stopwords=f.readlines()
    stopwords=[x.strip() for x in stopwords]
print(stopwords)
pos_dict = {}

def get_words(sen):
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
    return cutwords3


def word_cloud(path,alldata,name):
    object_list=[]
    data = alldata['item'].values
    for sen in tqdm(data):
        words=get_words(sen)
        for word in words: # 循环读出每个分词
            object_list.append(word) # 分词追加到列表

    # 词频统计
    word_counts = collections.Counter(object_list) # 对分词做词频统计
    word_counts_top20 = word_counts.most_common() # 获取词频
    allsum=np.sum([x[1] for x in word_counts_top20])

    wdpath=path+name+'_词频.csv'
    if os.path.exists(wdpath):
        os.remove(wdpath)
    for wd in word_counts_top20:
        wd_word=wd[0]
        wd_num=wd[1]
        wd_pos=pos_dict[wd_word]
        all_wd=wd_word,wd_num,wd_num/allsum,wd_pos
        with open(wdpath,'a',encoding='utf-8-sig',newline='')as f:
            writer=csv.writer(f)
            writer.writerow(all_wd)

    # 词频展示
    mask = np.array(Image.open('./data/back.jpg')) # 定义词频背景
    wc = wordcloud.WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式
        mask=mask, # 设置背景图
        #min_font_size=30,
        max_words=100, # 最多显示词数
        max_font_size=300 ,# 字体最大值
        background_color = "white"
    )

    wc.generate_from_frequencies(word_counts) # 从字典生成词云
    # image_colors = wordcloud.ImageColorGenerator(mask) # 从背景图建立颜色方案
    # wc.recolor(color_func=image_colors) # 将词云颜色设置为背景图方案
    plt.figure(figsize=(20,15))
    plt.imshow(wc) # 显示词云
    plt.axis('off') # 关闭坐标轴

    plt.savefig(path+'/'+name+'_word_cloud.jpg')
    # plt.show()  # 显示图像

#计算主题重要性
def cal_im(topics,topics_list):
    topics=list(topics)
    res=[]
    for x in topics:
        r=topics_list.count(x)
        r=r/len(topics_list)
        res.append(r)

    resdf=pd.DataFrame()
    resdf['主题']=topics
    resdf['重要性']=res
    resdf.to_csv('主题重要性.csv',index=None)


df=pd.read_csv('result.csv')
topics=set(df['max_Topic'].to_list())
topics_list=df['max_Topic'].to_list()
cal_im(topics,topics_list)

print(topics)
for t in topics:
    tdf=df[df['max_Topic']==t]
    print(tdf)
    name='topic_'+str(t)
    word_cloud('./res/',tdf,name)