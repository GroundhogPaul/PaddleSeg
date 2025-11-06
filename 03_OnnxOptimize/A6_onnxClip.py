import os
import onnx

# ----- read the ONNX model ----- #
sInputModel = "humansegv2_lite_192x192_with_softmax_simplified_opt.onnx"
sOutputModel = os.path.splitext(sInputModel)[0] + "_clip.onnx"
model = onnx.load(sInputModel)

# ----- output the -3 layer ----- #
graph = model.graph
nodes = graph.node

graph.node.remove(graph.node[-1])
graph.node.remove(graph.node[-1])
graph.node[-1].output[0] = "save_infer_model/scale_0.tmp_0"

print(graph.node[-1])

onnx.checker.check_model(model)
onnx.save(model, sOutputModel)