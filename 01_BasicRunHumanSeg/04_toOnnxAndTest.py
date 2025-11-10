import paddle2onnx
import os
import cv2
import numpy as np

# ----- file names ----- #
model_dir = r".\01_BasicRunHumanSeg"
assert os.path.exists(model_dir)
model_filename = os.path.join(model_dir, "model.pdmodel")
params_filename = os.path.join(model_dir, "model.pdiparams")
save_file = r".\01_BasicRunHumanSeg\humansegv2_lite_192x192_with_softmax.onnx"
opset_version = 11

# ----- export ----- #
paddle2onnx.export(
    model_filename = model_filename,
    params_filename = params_filename,
    save_file = save_file,
    opset_version = opset_version,
)

# ----- test the onnx exported ----- #
def preprocess_image_for_onnx(image_path, target_size=(192, 192)):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image_resized = cv2.resize(image, target_size)
    
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    image_normalized = image_rgb.astype(np.float32) / 255.0
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    
    image_batch = np.expand_dims(image_chw, axis=0)
    
    return image_batch, image

net = cv2.dnn.readNetFromONNX(save_file)
image_path = r"D:\users\xiaoyaopan\PxyAI\PaddleSeg\PaddleSeg\contrib\PP-HumanSeg\data\images\portrait_heng.jpg"
input_data, original_image = preprocess_image_for_onnx(image_path, (192, 192))
net.setInput(input_data)
outputs = net.forward()


squeezed_mat = np.squeeze(outputs)
reshaped_mat = np.expand_dims(squeezed_mat, axis=-1)
display_img = (reshaped_mat * 255).astype(np.uint8)
cv2.imshow('Reshaped Image', display_img)
cv2.waitKey(0)

print("done")