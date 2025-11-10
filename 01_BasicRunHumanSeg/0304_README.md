03_toInferAndAndTest.py

    Paddle训练后导出的模型分为训练用的checkpoint模型和推理用inference模型。

    checkpoint模型仅包含一个model.pdparams文件（模型参数，训练状态）

    inference模型包含四个文件。**`model.pdmodel`**（固化的模型结构），**`model.pdiparams`**（权重和偏置），**`model.pdiparams.info`**（各参数的名称，可选），**`deploy.yaml`**（格式和前后处理，可选）。

    其中后者要从前者转换出来。

    TODO：转换的方法还没试过，我是下载现成的。

04_toOnnxAndTest.py

    需要转到onnx的模块，必须是inference_model（也就是要先用03_toInferAndAndTest.py转换）

    具体过程详见 [飞桨模型转 ONNX 模型-使用文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/guides/advanced/model_to_onnx_cn.html)

    如果想直接使用现成的inference_model，可以在如下地址下载 [【PaddleSeg实践范例】PP-HumanSegV2 SOTA人像分割方案 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/4504982?contributionType=1)


