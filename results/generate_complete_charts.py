# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

lat_base  = np.load('C:/gpu_optimizer/results/base_latencies.npy')
lat_trt   = np.load('C:/gpu_optimizer/results/trt_latencies.npy')
lat_fp16  = np.load('C:/gpu_optimizer/results/trt_fp16_latencies.npy')
lat_comp  = np.load('C:/gpu_optimizer/results/lat_comp_clean.npy')
phase2    = np.load('C:/gpu_optimizer/results/phase2_times.npy')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('GPU Inference Optimization - Complete Results\nResNet-50 on GTX 1650 (4GB VRAM)',
             fontsize=14, fontweight='bold')

# Chart 1: Inference Latency
ax1 = axes[0,0]
methods = ['PyTorch\nBaseline', 'torch\n.compile', 'TensorRT\nFP32', 'TensorRT\nTF32']
avgs    = [lat_base.mean(), lat_comp.mean(), lat_trt.mean(), lat_fp16.mean()]
p99s    = [np.percentile(lat_base,99), np.percentile(lat_comp,99),
           np.percentile(lat_trt,99),  np.percentile(lat_fp16,99)]
colors  = ['#e74c3c','#f39c12','#27ae60','#2980b9']
x = np.arange(len(methods))
bars = ax1.bar(x, avgs, color=colors, alpha=0.85, width=0.5)
ax1.scatter(x, p99s, color='black', zorder=5, s=80, marker='D', label='99th %ile')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Inference Latency Comparison')
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.4, label='50ms deadline')
ax1.legend(fontsize=8)
for bar, avg in zip(bars, avgs):
    ax1.text(bar.get_x()+bar.get_width()/2, avg+0.1,
             f'{avg:.1f}ms', ha='center', fontsize=9, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 13)

# Chart 2: Preprocessing Speedup
ax2 = axes[0,1]
pre_methods = ['OpenCV\n(CPU)', 'PyTorch\nCUDA', 'Raw CUDA\nV1', 'Shared Mem\nV2']
pre_times   = [phase2[0], 0.595, phase2[1], phase2[2]]
pre_colors  = ['#e74c3c','#f39c12','#27ae60','#2980b9']
bars2 = ax2.bar(pre_methods, pre_times, color=pre_colors, alpha=0.85, width=0.5)
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Preprocessing Speedup (640x640 to 224x224)')
for bar, t in zip(bars2, pre_times):
    ax2.text(bar.get_x()+bar.get_width()/2, t+0.01,
             f'{t:.3f}ms', ha='center', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Chart 3: Latency Distribution
ax3 = axes[1,0]
ax3.hist(lat_base, bins=40, alpha=0.6, color='#e74c3c',
         label=f'Baseline (s={lat_base.std():.2f}ms)')
ax3.hist(lat_comp, bins=40, alpha=0.6, color='#f39c12',
         label=f'torch.compile (s={lat_comp.std():.2f}ms)')
ax3.hist(lat_trt,  bins=40, alpha=0.6, color='#27ae60',
         label=f'TensorRT FP32 (s={lat_trt.std():.2f}ms)')
ax3.set_xlabel('Latency (ms)')
ax3.set_ylabel('Count')
ax3.set_title('Latency Distribution (500 runs each)')
ax3.legend(fontsize=8)
ax3.set_xlim(0, 15)
ax3.grid(alpha=0.3)

# Chart 4: Summary Speedup
ax4 = axes[1,1]
summary_methods = ['torch.compile',
                   'TensorRT FP32',
                   'TRT 99th pct',
                   'CUDA Preprocess vs OpenCV',
                   'Shared Mem vs OpenCV']
speedups = [
    lat_base.mean()/lat_comp.mean(),
    lat_base.mean()/lat_trt.mean(),
    np.percentile(lat_base,99)/np.percentile(lat_trt,99),
    phase2[0]/phase2[1],
    phase2[0]/phase2[2]
]
colors4 = ['#f39c12','#27ae60','#2980b9','#8e44ad','#16a085']
bars4 = ax4.barh(summary_methods, speedups, color=colors4, alpha=0.85)
ax4.set_xlabel('Speedup (x)')
ax4.set_title('All Optimizations Summary')
ax4.axvline(x=1, color='red', linestyle='--', alpha=0.5)
for bar, spd in zip(bars4, speedups):
    ax4.text(spd+0.5, bar.get_y()+bar.get_height()/2,
             f'{spd:.2f}x', va='center', fontsize=9, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('C:/gpu_optimizer/results/complete_benchmark_charts.png',
            dpi=150, bbox_inches='tight')
print("Chart saved!")
plt.show()
