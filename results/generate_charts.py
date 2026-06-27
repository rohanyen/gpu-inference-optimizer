import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load results
lat_base  = np.load('C:/gpu_optimizer/results/lat_base_clean.npy')
lat_comp  = np.load('C:/gpu_optimizer/results/lat_comp_clean.npy')
lat_cudnn = np.load('C:/gpu_optimizer/results/lat_cudnn_clean.npy')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('GPU Inference Optimization Results\nResNet-50 on GTX 1650 (4GB VRAM)',
             fontsize=14, fontweight='bold')

# -- Chart 1: Latency Comparison --
ax1 = axes[0]
methods = ['FP32\nBaseline', 'cuDNN\nBenchmark', 'torch\n.compile']
avgs    = [lat_base.mean(), lat_cudnn.mean(), lat_comp.mean()]
p99s    = [np.percentile(lat_base,99), np.percentile(lat_cudnn,99), np.percentile(lat_comp,99)]
stds    = [lat_base.std(), lat_cudnn.std(), lat_comp.std()]
colors  = ['#e74c3c', '#f39c12', '#27ae60']
x = np.arange(len(methods))
bars = ax1.bar(x, avgs, color=colors, alpha=0.8, width=0.5, label='Average')
ax1.scatter(x, p99s, color='black', zorder=5, s=80, label='99th percentile', marker='D')
ax1.errorbar(x, avgs, yerr=stds, fmt='none', color='black', capsize=5, linewidth=2)
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Inference Latency Comparison')
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50ms deadline')
ax1.legend()
ax1.set_ylim(0, 35)
for bar, avg in zip(bars, avgs):
    ax1.text(bar.get_x() + bar.get_width()/2, avg + 0.5,
             f'{avg:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# -- Chart 2: Preprocessing Speedup --
ax2 = axes[1]
preprocess_methods = ['OpenCV\n(CPU)', 'PyTorch\nCUDA', 'Raw CUDA\nKernel']
preprocess_times   = [2.377, 0.595, 0.031]
preprocess_colors  = ['#e74c3c', '#f39c12', '#27ae60']
bars2 = ax2.bar(preprocess_methods, preprocess_times, color=preprocess_colors, alpha=0.8, width=0.5)
ax2.set_ylabel('Latency (ms)')
ax2.set_title('Preprocessing Speedup\n(640x640 ? 224x224 resize+normalize)')
for bar, t in zip(bars2, preprocess_times):
    ax2.text(bar.get_x() + bar.get_width()/2, t + 0.02,
             f'{t:.3f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
speedup_patch = mpatches.Patch(color='white', label='Raw CUDA: 76.6x vs OpenCV\nRaw CUDA: 19.2x vs PyTorch')
ax2.legend(handles=[speedup_patch], loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 3.0)

# -- Chart 3: Latency Distribution --
ax3 = axes[2]
ax3.hist(lat_base,  bins=50, alpha=0.6, color='#e74c3c', label=f'Baseline (s={lat_base.std():.2f}ms)')
ax3.hist(lat_cudnn, bins=50, alpha=0.6, color='#f39c12', label=f'cuDNN BM (s={lat_cudnn.std():.2f}ms)')
ax3.hist(lat_comp,  bins=50, alpha=0.6, color='#27ae60', label=f'torch.compile (s={lat_comp.std():.2f}ms)')
ax3.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50ms deadline')
ax3.set_xlabel('Latency (ms)')
ax3.set_ylabel('Count')
ax3.set_title('Latency Distribution\n(500 runs each)')
ax3.legend(fontsize=9)
ax3.set_xlim(0, 40)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('C:/gpu_optimizer/results/benchmark_charts.png', dpi=150, bbox_inches='tight')
print("Chart saved: C:/gpu_optimizer/results/benchmark_charts.png")
plt.show()
