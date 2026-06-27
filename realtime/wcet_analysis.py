# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*60)
print("WCET + JITTER ANALYSIS")
print("(Embedded Real-Time Thinking Applied to GPU)")
print("="*60)

# Load TensorRT engine (best performing)
logger  = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("C:/gpu_optimizer/tensorrt/resnet50_fp32.trt", 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

d_in  = cuda.mem_alloc(1*3*224*224*4)
d_out = cuda.mem_alloc(1*1000*4)
stream = cuda.Stream()
context.set_tensor_address('input',  int(d_in))
context.set_tensor_address('output', int(d_out))
h_in = np.random.randn(1,3,224,224).astype(np.float32)
cuda.memcpy_htod(d_in, h_in)

# Warmup
for _ in range(200):
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

# Collect 5000 samples for statistical analysis
N = 5000
print(f"Collecting {N} samples for WCET analysis...")
latencies = []
for i in range(N):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    latencies.append((time.perf_counter()-t0)*1000)
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{N} samples collected...")
latencies = np.array(latencies)

# WCET Analysis (Embedded Real-Time Concepts)
mean_lat  = latencies.mean()
std_lat   = latencies.std()
min_lat   = latencies.min()
max_lat   = latencies.max()
wcet      = latencies.max()  # Worst Case Execution Time
bcet      = latencies.min()  # Best Case Execution Time
jitter    = max_lat - min_lat  # Execution Time Jitter
p99_lat   = np.percentile(latencies, 99)
p999_lat  = np.percentile(latencies, 99.9)
p9999_lat = np.percentile(latencies, 99.99)

print(f"\n{'='*65}")
print(f"REAL-TIME ANALYSIS REPORT")
print(f"(Applying FreeRTOS/Embedded Thinking to GPU Inference)")
print(f"{'='*65}")
print(f"\nExecution Time Statistics ({N} samples):")
print(f"  BCET (Best Case):        {bcet:.3f} ms")
print(f"  Average:                 {mean_lat:.3f} ms")
print(f"  WCET (Worst Case):       {wcet:.3f} ms")
print(f"  Std Deviation:           {std_lat:.3f} ms")
print(f"  Jitter (WCET-BCET):      {jitter:.3f} ms")
print(f"\nPercentile Analysis:")
print(f"  50th percentile:         {np.percentile(latencies,50):.3f} ms")
print(f"  90th percentile:         {np.percentile(latencies,90):.3f} ms")
print(f"  99th percentile:         {p99_lat:.3f} ms")
print(f"  99.9th percentile:       {p999_lat:.3f} ms")
print(f"  99.99th percentile:      {p9999_lat:.3f} ms")

print(f"\nDeadline Compliance Analysis:")
deadlines = [5, 10, 20, 50, 100]
for d in deadlines:
    compliance = (latencies <= d).mean() * 100
    miss_rate  = 100 - compliance
    status = "HARD RT" if compliance == 100 else "SOFT RT" if compliance > 99 else "MISS"
    print(f"  {d:3d}ms deadline: {compliance:6.2f}% compliance "
          f"({miss_rate:.2f}% miss rate) [{status}]")

print(f"\nJitter Analysis:")
print(f"  Absolute jitter:         {jitter:.3f} ms")
print(f"  Relative jitter:         {jitter/mean_lat*100:.1f}% of mean")
print(f"  3-sigma bound:           {mean_lat + 3*std_lat:.3f} ms")
print(f"  6-sigma bound:           {mean_lat + 6*std_lat:.3f} ms")

print(f"\nEmbedded RT Comparison:")
print(f"  FreeRTOS task jitter:    ~0.001-0.01 ms (microseconds)")
print(f"  GPU inference jitter:    {jitter:.3f} ms")
print(f"  Ratio:                   {jitter/0.01:.0f}x more jitter than RTOS")
print(f"  BUT: GPU does 4.1 GFLOPs vs RTOS doing <1 MFLOP")
print(f"  Tradeoff: {jitter:.1f}ms jitter for 4100x more compute")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'WCET + Jitter Analysis — TensorRT on GTX 1650\n'
             f'({N} samples, Embedded Real-Time Perspective)',
             fontweight='bold')

# Chart 1: Time series
ax1 = axes[0,0]
ax1.plot(latencies[:500], alpha=0.7, color='#27ae60',
         linewidth=0.5, label='Latency')
ax1.axhline(y=mean_lat, color='blue', linestyle='--',
            label=f'Mean: {mean_lat:.2f}ms')
ax1.axhline(y=wcet, color='red', linestyle='-',
            label=f'WCET: {wcet:.2f}ms', linewidth=2)
ax1.axhline(y=bcet, color='green', linestyle='-',
            label=f'BCET: {bcet:.2f}ms', linewidth=2)
ax1.fill_between(range(500), bcet, wcet, alpha=0.1, color='yellow',
                 label=f'Jitter zone: {jitter:.2f}ms')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Execution Time over 500 Samples')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Chart 2: Distribution
ax2 = axes[0,1]
ax2.hist(latencies, bins=100, color='#3498db', alpha=0.8,
         edgecolor='white', density=True)
ax2.axvline(x=mean_lat, color='blue', linestyle='--',
            label=f'Mean: {mean_lat:.2f}ms')
ax2.axvline(x=wcet, color='red', linestyle='-',
            label=f'WCET: {wcet:.2f}ms', linewidth=2)
ax2.axvline(x=p99_lat, color='orange', linestyle='--',
            label=f'99p: {p99_lat:.2f}ms')
ax2.set_xlabel('Latency (ms)')
ax2.set_ylabel('Density')
ax2.set_title('Latency Distribution (Density)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Chart 3: Percentile analysis
ax3 = axes[1,0]
pcts = [50, 75, 90, 95, 99, 99.9, 99.99]
vals = [np.percentile(latencies, p) for p in pcts]
ax3.plot([str(p) for p in pcts], vals, 'bo-',
         linewidth=2, markersize=8)
ax3.set_xlabel('Percentile (%)')
ax3.set_ylabel('Latency (ms)')
ax3.set_title('Percentile Analysis')
ax3.grid(alpha=0.3)
for x, y in zip([str(p) for p in pcts], vals):
    ax3.annotate(f'{y:.2f}ms', (x, y),
                 textcoords="offset points",
                 xytext=(0, 8), ha='center', fontsize=8)

# Chart 4: Deadline compliance
ax4 = axes[1,1]
deadlines_plot = [5, 10, 20, 50, 100]
compliance_plot = [(latencies <= d).mean()*100 for d in deadlines_plot]
bars = ax4.bar([str(d)+'ms' for d in deadlines_plot],
               compliance_plot,
               color=['#e74c3c' if c < 100 else '#27ae60'
                      for c in compliance_plot],
               alpha=0.85)
ax4.set_xlabel('Deadline')
ax4.set_ylabel('Compliance (%)')
ax4.set_title('Deadline Compliance Analysis')
ax4.set_ylim(95, 101)
ax4.axhline(y=100, color='green', linestyle='--', alpha=0.5)
for bar, c in zip(bars, compliance_plot):
    ax4.text(bar.get_x()+bar.get_width()/2, c+0.05,
             f'{c:.2f}%', ha='center', fontsize=9, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('C:/gpu_optimizer/results/wcet_jitter_analysis.png',
            dpi=150, bbox_inches='tight')
print("\nChart saved: wcet_jitter_analysis.png")
plt.show()

np.save('C:/gpu_optimizer/results/wcet_latencies.npy', latencies)
print("Results saved.")
