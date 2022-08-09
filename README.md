# intern-LDA-model

## huaban-scraping.py
1. 使用selenium等模组爬取图库网站中的图片信息
2. 运行前请安装Chrome浏览器的webdriver
3. 请自行填写相关的网站登录信息
4. 爬取图片的数量由“下拉至窗口底部的次数”决定：每10次下拉可爬取大约200个图片的信息
5. 结果储存在result.csv中

## ciyun_generate.py
1. 使用wordcloud等模组来生成词云
2. 使用了存储在result.csv中图片的标签来生成词云
3. 可以选择向词云中加入蒙版（mask），生成指定形状的词云，并且词云图的颜色可从蒙版里提取
4. 若想使用蒙版，则需将蒙版图片改名为background.jpg并放在根目录下（或更改py文件中的路径）

## LDA_model_generate.py
1. 使用gensim等模组构建针对result.csv中图片标签的LDA模型
2. 对数据预处理时使用的停用词表存放在stopword.txt中
3. 可根据个人选择更改生成LDA模型的参数
4. 最后LDA模型的可视化会存入LDA_Visualization.html，可以通过live server等插件在浏览器中打开