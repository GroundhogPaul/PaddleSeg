import os
import onnx
from onnxsim import simplify

# ----- read the ONNX model ----- #
sInputModel = "humansegv2_lite_192x192_with_softmax.onnx"
sOutputModel = os.path.splitext(sInputModel)[0] + "_simplified.onnx"
model = onnx.load(sInputModel)

# ----- Simplify the ONNX model ----- #
model_sim, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"

# ----- output the simplified ONNX model ----- #
onnx.save(model_sim, sOutputModel)
print(f"Simplified model saved to: {sOutputModel}")