# ---------------------------------导入模块------------------------------
import jieba
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models, similarities
import re, string
from zhon.hanzi import punctuation
import seaborn as sns
import matplotlib.colors as mcolors

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ------------------------------对获取的数据进行分词-------------------------------
# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('stopword.txt', 'r', encoding='UTF-8').readlines()]
    return stopwords

# 定义停词函数 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

# 寻找LDA的最佳主题数，并返还使用其作为参数来构建的最佳模型
def find_optimal_model(corpus, id2word, result_fenci):
    coherence_values = []
    model_list = []
    for num_topics in range(2,41,2):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                        id2word=id2word,
                        num_topics=num_topics,
                        # random_state=100,
                        # update_every=1,
                        # chunksize=100,
                        # passes=10,
                        # alpha='auto',
                        # per_word_topics=True
    )
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(model=lda_model, texts=result_fenci, dictionary=id2word, coherence='u_mass')
        coherence_values.append(round(coherencemodel.get_coherence(),3))

    # 最佳主题数可视化展示
    # 利用可视化的方式进行展示
    x = range(2,41,2)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('最佳主题数可视化展示:')
    plt.show()
    print('展示结束')

    # 展示不同主题数的一致性分数
    # 展示不同的主题数对应的一致性分数
    print('--------------------------------------------------------------------------------------------------------------------------')
    print('不同的主题数对应的一致性分数:')
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    print('--------------------------------------------------------------------------------------------------------------------------')
    # LDA模型中的最佳主题
    # 根据一致性得分，选择具有最高CV的模型，选择的主题数是
    max_cv = max(coherence_values)
    index = coherence_values.index(max_cv)
    best_topic_num = x[index]
    print(f'最高一致性得分为:{max_cv}')
    print(f'最佳主题数为:{best_topic_num}')
    # 选择模型并打印主题
    optimal_model = model_list[index]
    model_topics = optimal_model.show_topics(formatted=False)
    # print('使用最佳主题数产生的模型的主题为:')
    # pprint(optimal_model.print_topics(num_topics=-1, num_words=20))

    return optimal_model, best_topic_num

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    # print(contents)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    # print(sent_topics_df)
    return(sent_topics_df)

# ----------------------------------获取数据------------------------------------
fp = pd.read_csv('result.csv', encoding='utf-8') 
text = list(fp.tag)  # 获取标签名
# print(punctuation)
for index, item in enumerate(text):
    # 去除标签中空格
    text[index] = text[index].replace(' ', '')
    # 去除标签中英文标点
    for i in string.punctuation:
        text[index] = text[index].replace(i, '')
    for j in punctuation:
        text[index] = text[index].replace(j, '') 
# print('-------------------------------------------------------------')
# print('标签列表:')
# print(text)

# 分词后的结果
result_fenci = []
for i in text:
    # print(i)
    if seg_depart(i) != '':
        # print(seg_depart(i))
        result_fenci.append([i, seg_depart(i)])
result_fenci = [i[1].split(' ')[:-1] for i in result_fenci]
# # pd.DataFrame(result_fenci,columns=['rawtext','fencitext']).to_excel(path+'result.xlsx',index=False)

# 不分词
# result_fenci = []
# for i in text:
#     result_fenci.append([i])

# 分词结果展示

print('-------------------------------------------------------------')
print('将每个图片信息视作一个文档, 每个图片的标签视作这个文档中的文本')
print('分词结果为')
print(result_fenci)

# Create Dictionary
id2word = corpora.Dictionary(result_fenci)

# 创建字典和词袋(corpus)
# 将字典转换为词袋,为文档中的每一个单词创建唯一的ID
corpus = [id2word.doc2bow(sentence) for sentence in result_fenci]
# print('--------------------------------------------------------------------------------------------------------------------------')
# print('列表套子列表，子列表中套元组:')
# print(corpus[:1])
# print(id2word)
# print(corpus)
# [[(1,3),(2,1)],[]] 返回的结果是列表套子列表，子列表中套元组，如（1,3）代表1在第一个文档中出现3次。

# 可通过如下预期查询id对应的词
# print('--------------------------------------------------------------------------------------------------------------------------')
# print('不同id对应的词:')
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

# -------------------------------构建主题模型-----------------------------------
'''
在建立主题模型时，除了字典和语料库之外，我们还需要对模型的参数，如主题数据量等
num_topics代表生成的主题数
chunksize是每个训练块中使用的文档数;
update_every确定应更新模型参数的频率;
passes培训通过的总数;
'''
# 建立LDA模型
lda_model, best_topic_num  = find_optimal_model(corpus, id2word, result_fenci)

# 上述LDA模型由10个不同的主题构建，其中每个主题是关键字的组合，并且每个关键字对主题贡献一定的权重，权重反应了关键字对主体的贡献程度。
# num_word代表每个主题的关键字数
print('--------------------------------------------------------------------------------------------------------------------------')
print(f'将LDA模型分解成{best_topic_num}个不同的主题构建: 其中每个主题是关键字的组合，并且每个关键字对主题贡献一定的权重，权重反应了关键字对主体的贡献程度:')
pprint(lda_model.print_topics(num_topics=-1, num_words=20))

'''
模型复杂度和主题一致性提供了一种方便的方法来判断给定主题模型的好坏程度。
特别是主题一致性得分更有帮助。
'''
# def model():
# Compute Perplexity 模型复杂度
print('--------------------------------------------------------------------------------------------------------------------------')
print('模型复杂度:')
print('Perplexity: ', lda_model.log_perplexity(corpus)) # a measure of how good the model is. lower the better.

# Compute Coherence Score 主题一致性
coherence_model_lda = CoherenceModel(model=lda_model, texts=result_fenci, dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('--------------------------------------------------------------------------------------------------------------------------')
print('主题一致性得分:')
print('Coherence Score: ', coherence_lda) # 主题一致性得分

# -----------------------------------------模型可视化-----------------------------------------
# 模型可视化：将结果存入HTML文件中，可通过live server打开
# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')

# ----------------------------------------在每个标签中找到主导主题--------------------------------------------
# 每个文档（图片的标签中）都包含多个主题。但是，通常只有一个主题是主导的。下面的代码为每个句子提取该主导主题，并在格式正确的输出中显示该主题和关键字的权重。
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=result_fenci)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.to_excel('每个图片的主导主题.xlsx',index=False)
df_dominant_topic.head(10)

# -----------------------------------------每个主题中最具有代表性的标签-----------------------------------------
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.to_excel('每个主题最具代表性的标签.xlsx',index=False)
sent_topics_sorteddf_mallet.head(10)

# -----------------------------------------各个图片标签中字数的频率分布-----------------------------------------
df_dominant_topic = df_dominant_topic.dropna(axis=0)
doc_lens = [len(d) for d in df_dominant_topic.Text]

# Plot
plt.figure(figsize=(8,5), dpi=160)
plt.hist(doc_lens, bins = 20, color='navy')
plt.text(750, 100, "Mean : " + str(round(np.mean(doc_lens))))
plt.text(750, 90, "Median : " + str(round(np.median(doc_lens))))
plt.text(750, 80, "Stdev : " + str(round(np.std(doc_lens))))
plt.text(750, 70, "1%ile : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(750, 60, "99%ile : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 20), ylabel='Number of Images', xlabel='Image Label Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,20,9))
plt.title('Distribution of Image Label Word Counts', fontdict=dict(size=12))
print('--------------------------------------------------------------------------------------------------------------------------')
print('各个图片标签中字数的频率分布展示:')
plt.show()
print('展示结束')

# -----------------------------------------各个主题中图片标签字数的频率分布----------------------------------------
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] # more colors: 'mcolors.XKCD_COLORS'

fig, axes = plt.subplots(2,2,figsize=(8,7), dpi=160, sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 20, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 20), xlabel='Image Label Word Count')
    ax.set_ylabel('Number of Images', color=cols[i])
    ax.set_title('Topic: '+str(i), fontdict=dict(size=10, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,20,9))
fig.suptitle('Distribution of Image Label Word Counts by Dominant Topic', fontsize=12)
print('--------------------------------------------------------------------------------------------------------------------------')
print('各个主题中图片标签字数的频率分布展示:')
plt.show()
print('展示结束')
print('--------------------------------------------------------------------------------------------------------------------------')