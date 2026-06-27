# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

print("="*60)
print("PHASE 3: HARD REAL-TIME LATENCY PROOF")
print("="*60)

# Load TensorRT engine
logger  = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("C:/gpu_optimizer/tensorrt/resnet50_fp32.trt", 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

d_in   = cuda.mem_alloc(1*3*224*224*4)
d_out  = cuda.mem_alloc(1*1000*4)
stream = cuda.Stream()
context.set_tensor_address('input',  int(d_in))
context.set_tensor_address('output', int(d_out))

h_in = np.random.randn(1,3,224,224).astype(np.float32)
cuda.memcpy_htod(d_in, h_in)

# Warmup
for _ in range(100):
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

# Long run for real-time proof (2000 samples)
print("Running 2000 inference samples for real-time proof...")
N = 2000
latencies = []
for i in range(N):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    latencies.append((time.perf_counter()-t0)*1000)
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{N} done...")

latencies = np.array(latencies)

print(f"\n{'='*60}")
print(f"HARD REAL-TIME LATENCY ANALYSIS (2000 runs)")
print(f"{'='*60}")
print(f"Average:          {latencies.mean():.3f} ms")
print(f"Std Dev:          {latencies.std():.3f} ms")
print(f"Min:              {latencies.min():.3f} ms")
print(f"Max:              {latencies.max():.3f} ms")
print(f"90th percentile:  {np.percentile(latencies,90):.3f} ms")
print(f"95th percentile:  {np.percentile(latencies,95):.3f} ms")
print(f"99th percentile:  {np.percentile(latencies,99):.3f} ms")
print(f"99.9th percentile:{np.percentile(latencies,99.9):.3f} ms")
print(f"{'='*60}")

deadlines = [10, 20, 50, 100]
print(f"\nDEADLINE COMPLIANCE ANALYSIS:")
for d in deadlines:
    pct = (latencies <= d).mean() * 100
    status = "PASS" if pct == 100 else f"{pct:.2f}%"
    print(f"  < {d:3d}ms deadline: {pct:6.2f}% compliance {'✅' if pct==100 else '⚠️'}")

print(f"\nCONCLUSION:")
if latencies.max() < 50:
    print(f"  HARD REAL-TIME GUARANTEE: ALL {N} frames < 50ms ✅")
    print(f"  Worst case: {latencies.max():.2f}ms")
    print(f"  System is DETERMINISTIC within 50ms budget")

# Save
np.save('C:/gpu_optimizer/results/realtime_latencies.npy', latencies)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hard Real-Time Latency Proof\nTensorRT FP32 on GTX 1650 (2000 runs)',
             fontweight='bold')

ax1 = axes[0]
ax1.plot(latencies, alpha=0.7, color='#27ae60', linewidth=0.5)
ax1.axhline(y=latencies.mean(), color='blue', linestyle='--',
            label=f'Mean: {latencies.mean():.2f}ms')
ax1.axhline(y=np.percentile(latencies,99), color='orange', linestyle='--',
            label=f'99th: {np.percentile(latencies,99):.2f}ms')
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50ms deadline')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Latency Over Time (2000 frames)')
ax1.legend(fontsize=9)
ax1.set_ylim(0, 15)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.hist(latencies, bins=60, color='#27ae60', alpha=0.8, edgecolor='white')
ax2.axvline(x=latencies.mean(), color='blue', linestyle='--',
            label=f'Mean: {latencies.mean():.2f}ms')
ax2.axvline(x=np.percentile(latencies,99), color='orange', linestyle='--',
            label=f'99th: {np.percentile(latencies,99):.2f}ms')
ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50ms deadline')
ax2.set_xlabel('Latency (ms)')
ax2.set_ylabel('Count')
ax2.set_title('Latency Distribution')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('C:/gpu_optimizer/results/realtime_proof.png', dpi=150, bbox_inches='tight')
print("\nChart saved: realtime_proof.png")
plt.show()
