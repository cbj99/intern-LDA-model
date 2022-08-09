import matplotlib.pyplot as plt #数据可视化
import jieba #词语切割
import wordcloud #分词
import pandas as pd
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS #词云，颜色生成器，停止词
import numpy as np #科学计算
from PIL import Image #处理图片
from collections import Counter

def ciyun():
    fp = pd.read_csv('result.csv', encoding='utf-8')  # 读取词频文件, 因为要显示中文，故编码为gbk
    tag_names = list(fp.tag)  # 标签名
    # print(tag_names)
    tag_frequency = Counter(tag_names)
    # print(tag_frequency)
    
    backgroud = np.array(Image.open('background.jpg')) 
	
    wc = WordCloud(width=1400, height=2200,
			background_color='white',
	        mode='RGB', 
			# mask=backgroud, #添加蒙版，生成指定形状的词云，并且词云图的颜色可从蒙版里提取
			max_words=500,
			# stopwords=STOPWORDS.add('老年人'),#内置的屏蔽词,并添加自己设置的词语
			font_path='C:\Windows\Fonts\STZHONGS.ttf',
			max_font_size=150,
			relative_scaling=0.6, #设置字体大小与词频的关联程度为0.4
			random_state=50, 
			scale=2 
			).generate_from_frequencies(tag_frequency)
			
    # image_color = ImageColorGenerator(backgroud)#设置生成词云的颜色，如去掉这两行则字体为默认颜色
    # wc.recolor(color_func=image_color)
	
    plt.imshow(wc) #显示词云
    plt.axis('off') #关闭x,y轴
    plt.show()#显示
    wc.to_file('ciyun.jpg') #保存词云图
    # wc.to_file('ciyun_with_background.jpg') #保存词云图

def main():
    ciyun()
 
if __name__ == '__main__':
    main()