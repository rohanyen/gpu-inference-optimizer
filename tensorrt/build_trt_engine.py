import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import os

print("="*60)
print("PHASE 2: TENSORRT OPTIMIZATION (TRT 11 API)")
print("="*60)
print(f"TensorRT version: {trt.__version__}")

# Step 1: Export to ONNX
print("\nStep 1: Exporting ResNet-50 to ONNX...")
model = models.resnet50(weights='DEFAULT').cuda().eval()
dummy = torch.randn(1, 3, 224, 224).cuda()

onnx_path = "C:/gpu_optimizer/tensorrt/resnet50.onnx"
torch.onnx.export(
    model, dummy, onnx_path,
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print(f"ONNX saved: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")

# Step 2: Build TensorRT engine (TRT 11 API)
print("\nStep 2: Building TensorRT engine (TRT 11 API)...")
logger  = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)

# TRT 11: no flag needed for EXPLICIT_BATCH
network = builder.create_network()
parser  = trt.OnnxParser(network, logger)

with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)

print("Building FP32 engine (1-2 mins)...")
serialized = builder.build_serialized_network(network, config)

engine_path = "C:/gpu_optimizer/tensorrt/resnet50_fp32.trt"
with open(engine_path, 'wb') as f:
    f.write(serialized)
print(f"Engine saved! Size: {os.path.getsize(engine_path)/1024/1024:.1f} MB")
print("TensorRT FP32 engine built successfully!")
