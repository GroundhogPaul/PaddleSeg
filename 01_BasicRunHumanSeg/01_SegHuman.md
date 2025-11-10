## 下载测试数据和模型

there are 2 src to download dataset and model

    分别是 [使用PP-HumanSegV2进行人像分割](https://aistudio.baidu.com/projectdetail/4504982?contributionType=1) 下的段落3.2和 2.2

## 运行测试代码

使用launch.json中的对应选项运行文件"./contrib/PP-HumanSeg/src/seg_demo.py"

## 学习后处理

人像分割完成后的两步后处理中的第一步

代码位置：“./contrib/PP-HumanSeg/src/infer.py"的Predictor.postporcess()

功能：用相对小的erode去掉小的孤立块，用相对大的dilate填补相对大空洞

## 学习光流法

人像分割完成后的两步后处理中的第二步
