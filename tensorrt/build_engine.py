import torch
import torchvision.models as models
import numpy as np
import tensorrt as trt
import os

print("="*50)
print("TENSORRT OPTIMIZATION")
print("="*50)

# Step 1: Export model to ONNX
print("\nStep 1: Exporting ResNet-50 to ONNX...")
model = models.resnet50(weights='DEFAULT').cuda().eval()
dummy = torch.randn(1, 3, 224, 224).cuda()

onnx_path = "C:/gpu_optimizer/tensorrt/resnet50.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
print(f"ONNX model saved: {onnx_path}")
print(f"ONNX file size: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")

# Step 2: Build TensorRT engine
print("\nStep 2: Building TensorRT FP32 engine...")
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

print("Building engine (this takes 1-2 minutes)...")
serialized_engine = builder.build_serialized_network(network, config)

engine_path = "C:/gpu_optimizer/tensorrt/resnet50_fp32.trt"
with open(engine_path, 'wb') as f:
    f.write(serialized_engine)
print(f"TensorRT engine saved: {engine_path}")
print(f"Engine size: {os.path.getsize(engine_path)/1024/1024:.1f} MB")
print("\nDone. Run trt_inference.py next to benchmark.")
