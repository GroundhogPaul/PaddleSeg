功能：

    将paddles seg的预训练模型转换成可用的onnx

输入：

    输入的神经网络：来自[【PaddleSeg实践范例】PP-HumanSegV2 SOTA人像分割方案 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/4504982?contributionType=1)

    注意，要下载其中的Checkpoint，而不是Inference Model

代码：

    代码完全照抄了 tools/export.py，只是把命令行参数parse_args改成了代码中的参数argparse.Namespace。



# 问题1：Ckpt和部署模型

    PaddleSeg有两种方式提供预训练模型，Ckpt（model.pdparams，静态图完整字典）和部署模型（model.pdiparams，仅含Tensor），其中的.pdparams用tools/export.py代码转换，而model.pdiparams用paddle2onnx工具转换。

    但用paddle2onnx工具转换model.pdiparams，涉及到Tensor回转成静态图再转成onnx，且工具年久失修，容易出问题。现在遇到的最直接的问题就是，转出来的onnx不知道怎么设置成固定size输入，导致onnx中有很多动态参数，opencv和onnxruntime都不支持。
