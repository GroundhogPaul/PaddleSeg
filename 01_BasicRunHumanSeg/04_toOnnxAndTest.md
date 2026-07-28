经过03_ExportPdi.py的导出，会得到

    inference模型包含四个文件。**`model.pdmodel`**（固化的模型结构），**`model.pdiparams`**（权重和偏置），**`model.pdiparams.info`**（各参数的名称，可选），**`deploy.yaml`**（格式和前后处理，可选）。

    

用paddle2onnx导出成onnx，产生的onnx用netron应该能看到固定size

用OpenCV-python载入，可以推理出图像
