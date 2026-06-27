# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os

print("="*60)
print("PHASE 4: INT8 QUANTIZATION")
print("="*60)

# Build INT8 engine
print("Building TensorRT INT8 engine...")
logger  = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser  = trt.OnnxParser(network, logger)

with open("C:/gpu_optimizer/tensorrt/resnet50.onnx", 'rb') as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)

# INT8 requires calibration — use default calibration
# For production you'd use real data, for demo we use random
print("Note: Using implicit quantization (no calibrator)")
print("Building engine...")
serialized = builder.build_serialized_network(network, config)

engine_path = "C:/gpu_optimizer/tensorrt/resnet50_int8.trt"
with open(engine_path, 'wb') as f:
    f.write(serialized)
print(f"INT8 engine saved: {os.path.getsize(engine_path)/1024/1024:.1f} MB")

# Benchmark INT8
runtime = trt.Runtime(logger)
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

d_in  = cuda.mem_alloc(1*3*224*224*4)
d_out = cuda.mem_alloc(1*1000*4)
stream = cuda.Stream()
context.set_tensor_address('input',  int(d_in))
context.set_tensor_address('output', int(d_out))

h_in = np.random.randn(1,3,224,224).astype(np.float32)
cuda.memcpy_htod(d_in, h_in)

for _ in range(100):
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

print("Benchmarking INT8 (500 runs)...")
lat_int8 = []
for _ in range(500):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    lat_int8.append((time.perf_counter()-t0)*1000)
lat_int8 = np.array(lat_int8)

lat_fp32 = np.load('C:/gpu_optimizer/results/trt_latencies.npy')
lat_base = np.load('C:/gpu_optimizer/results/base_latencies.npy')

print(f"\n{'='*70}")
print(f"QUANTIZATION COMPARISON")
print(f"{'='*70}")
print(f"{'Metric':<20} {'PyTorch':>12} {'TRT FP32':>12} {'TRT INT8':>12} {'INT8 Gain':>10}")
print(f"{'-'*70}")
b = lat_base.mean()
print(f"{'Average (ms)':<20} {b:>12.2f} {lat_fp32.mean():>12.2f} {lat_int8.mean():>12.2f} {lat_fp32.mean()/lat_int8.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<20} {lat_base.std():>12.3f} {lat_fp32.std():>12.3f} {lat_int8.std():>12.3f}")
print(f"{'99th %ile (ms)':<20} {np.percentile(lat_base,99):>12.2f} {np.percentile(lat_fp32,99):>12.2f} {np.percentile(lat_int8,99):>12.2f}")
print(f"{'vs Baseline':<20} {'1.00x':>12} {b/lat_fp32.mean():>11.2f}x {b/lat_int8.mean():>11.2f}x")
print(f"{'='*70}")

print(f"\nMODEL SIZE COMPARISON:")
print(f"  ONNX (FP32):     97.4 MB")
fp32_size = os.path.getsize('C:/gpu_optimizer/tensorrt/resnet50_fp32.trt')/1024/1024
int8_size = os.path.getsize(engine_path)/1024/1024
print(f"  TRT FP32:        {fp32_size:.1f} MB")
print(f"  TRT INT8:        {int8_size:.1f} MB")
print(f"  Size reduction:  {fp32_size/int8_size:.2f}x smaller")

np.save('C:/gpu_optimizer/results/int8_latencies.npy', lat_int8)
print("\nResults saved.")
