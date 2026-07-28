功能：

    将paddles seg的预训练模型从dict文件（.pdparams）转换成paddle框架的部署文件（model.pdiparams），为后续转onnx做准备

输入：

    输入的神经网络：来自[【PaddleSeg实践范例】PP-HumanSegV2 SOTA人像分割方案 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/4504982?contributionType=1)

    注意，要下载其中的Checkpoint，而不是Inference Model

代码：

    代码完全照抄了 tools/export.py，只是把命令行参数parse_args改成了代码中的参数argparse.Namespace。

# 问题1：Ckpt和部署模型

    PaddleSeg有两种方式提供预训练模型，Ckpt（model.pdparams，静态图完整字典）和部署模型（model.pdiparams，仅含Tensor）。其中的.pdparams用tools/export.py代码转换成pdiparams，而pdiparams可以用paddle2onnx工具转换成onnx。

    但PaddleSeg提供的部署模型.pdiparams往往不是固定size输入的，转换出的onnx也就不是固定size输入的，导致onnx中有很多动态参数，opencv和onnxruntime都不支持。

    所以要用代码中export从.pdparams转成.pdiparams，此时可以添加固定size输入。

# 问题2：模型某些算子，Opencv不支持

    为了方便跑通，找到一个路子，用[【PaddleSeg实践范例】PP-HumanSegV2 SOTA人像分割方案 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/4504982?contributionType=1)的pretrain模型（一个大小为12,227,618 字节的model.pdparams文件），作为03_ExportPdi.py的输入
