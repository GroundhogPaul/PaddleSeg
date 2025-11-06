import os
import onnx
from onnxoptimizer import optimize

# ----- read the ONNX model ----- #
sInputModel = "humansegv2_lite_192x192_with_softmax_simplified.onnx"
sOutputModel = os.path.splitext(sInputModel)[0] + "_opt.onnx"
model = onnx.load(sInputModel)

# ----- Optimize the simplified ONNX model ----- #
model_opt = optimize(model)

# ----- output the optimized ONNX model ----- #
onnx.save(model_opt, sOutputModel)
print(f"Optimized model saved to: {sOutputModel}")