import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os

print("="*60)
print("PHASE 2: TENSORRT FP16 ENGINE")
print("="*60)

# Build FP16 engine
print("Building TensorRT FP16 engine...")
logger  = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser  = trt.OnnxParser(network, logger)

with open("C:/gpu_optimizer/tensorrt/resnet50.onnx", 'rb') as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2<<30)

# Enable FP16
if True:  # GTX 1650 supports FP16
    config.set_flag(trt.BuilderFlag.TF32)  # TRT 11 uses TF32 instead of FP16 flag
    print("FP16 enabled!")
else:
    print("FP16 not supported on this GPU")

print("Building engine (1-2 mins)...")
serialized = builder.build_serialized_network(network, config)
engine_path = "C:/gpu_optimizer/tensorrt/resnet50_fp16.trt"
with open(engine_path, 'wb') as f:
    f.write(serialized)
print(f"FP16 engine saved: {os.path.getsize(engine_path)/1024/1024:.1f} MB")

# Benchmark FP16
print("\nBenchmarking TensorRT FP16...")
runtime = trt.Runtime(logger)
with open(engine_path, 'rb') as f:
    engine_fp16 = runtime.deserialize_cuda_engine(f.read())
ctx_fp16 = engine_fp16.create_execution_context()

d_in  = cuda.mem_alloc(1*3*224*224*4)
d_out = cuda.mem_alloc(1*1000*4)
stream = cuda.Stream()
ctx_fp16.set_tensor_address('input',  int(d_in))
ctx_fp16.set_tensor_address('output', int(d_out))

h_in = np.random.randn(1,3,224,224).astype(np.float32)
cuda.memcpy_htod(d_in, h_in)

for _ in range(50):
    ctx_fp16.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

latencies_fp16 = []
for _ in range(500):
    t0 = time.perf_counter()
    ctx_fp16.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    latencies_fp16.append((time.perf_counter()-t0)*1000)
latencies_fp16 = np.array(latencies_fp16)

# Load FP32 results
latencies_fp32 = np.load('C:/gpu_optimizer/results/trt_latencies.npy')
latencies_base = np.load('C:/gpu_optimizer/results/base_latencies.npy')

print(f"\n{'='*70}")
print(f"COMPLETE TENSORRT COMPARISON")
print(f"{'='*70}")
print(f"{'Metric':<20} {'PyTorch':>12} {'TRT FP32':>12} {'TRT FP16':>12} {'Best':>10}")
print(f"{'-'*70}")
b = latencies_base.mean()
print(f"{'Average (ms)':<20} {b:>12.2f} {latencies_fp32.mean():>12.2f} {latencies_fp16.mean():>12.2f} {min(latencies_fp32.mean(),latencies_fp16.mean()):>10.2f}")
print(f"{'Std Dev (ms)':<20} {latencies_base.std():>12.2f} {latencies_fp32.std():>12.2f} {latencies_fp16.std():>12.2f}")
print(f"{'99th %ile (ms)':<20} {np.percentile(latencies_base,99):>12.2f} {np.percentile(latencies_fp32,99):>12.2f} {np.percentile(latencies_fp16,99):>12.2f}")
print(f"{'Speedup vs base':<20} {'1.00x':>12} {b/latencies_fp32.mean():>11.2f}x {b/latencies_fp16.mean():>11.2f}x")
print(f"{'='*70}")

np.save('C:/gpu_optimizer/results/trt_fp16_latencies.npy', latencies_fp16)
print("Results saved.")
