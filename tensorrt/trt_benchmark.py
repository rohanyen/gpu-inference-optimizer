import torch
import torchvision.models as models
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time

print("="*60)
print("PHASE 2: TENSORRT BENCHMARK (TRT 11 API)")
print("="*60)

# Load engine
logger  = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("C:/gpu_optimizer/tensorrt/resnet50_fp32.trt", 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
print("Engine loaded!")

# TRT 11 API - use tensor addresses directly
input_shape  = (1, 3, 224, 224)
output_shape = (1, 1000)

d_input  = cuda.mem_alloc(int(np.prod(input_shape))  * 4)
d_output = cuda.mem_alloc(int(np.prod(output_shape)) * 4)
stream   = cuda.Stream()

# Set tensor addresses (TRT 11 way)
context.set_tensor_address('input',  int(d_input))
context.set_tensor_address('output', int(d_output))

# Prepare input
h_input = np.random.randn(*input_shape).astype(np.float32)
cuda.memcpy_htod(d_input, h_input)

def trt_infer():
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

# Warmup
print("Warming up TensorRT...")
for _ in range(50):
    trt_infer()

# Benchmark TensorRT
print("Benchmarking TensorRT FP32 (500 runs)...")
latencies_trt = []
for _ in range(500):
    t0 = time.perf_counter()
    trt_infer()
    latencies_trt.append((time.perf_counter()-t0)*1000)
latencies_trt = np.array(latencies_trt)

# Benchmark PyTorch baseline
print("Benchmarking PyTorch baseline (500 runs)...")
model = models.resnet50(weights='DEFAULT').cuda().eval()
dummy = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    for _ in range(50): _ = model(dummy)
torch.cuda.synchronize()

latencies_base = []
with torch.no_grad():
    for _ in range(500):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); _ = model(dummy); e.record()
        torch.cuda.synchronize()
        latencies_base.append(s.elapsed_time(e))
latencies_base = np.array(latencies_base)

print(f"\n{'='*65}")
print(f"TENSORRT FP32 vs PyTorch BASELINE")
print(f"{'='*65}")
print(f"{'Metric':<20} {'PyTorch':>15} {'TensorRT':>15} {'Speedup':>10}")
print(f"{'-'*65}")
b = latencies_base.mean()
print(f"{'Average (ms)':<20} {b:>15.2f} {latencies_trt.mean():>15.2f} {b/latencies_trt.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<20} {latencies_base.std():>15.2f} {latencies_trt.std():>15.2f}")
print(f"{'Min (ms)':<20} {latencies_base.min():>15.2f} {latencies_trt.min():>15.2f}")
print(f"{'Max (ms)':<20} {latencies_base.max():>15.2f} {latencies_trt.max():>15.2f}")
print(f"{'99th %ile (ms)':<20} {np.percentile(latencies_base,99):>15.2f} {np.percentile(latencies_trt,99):>15.2f} {np.percentile(latencies_base,99)/np.percentile(latencies_trt,99):>9.2f}x")
print(f"{'='*65}")

np.save('C:/gpu_optimizer/results/trt_latencies.npy',  latencies_trt)
np.save('C:/gpu_optimizer/results/base_latencies.npy', latencies_base)
print("Results saved.")
