# -*- coding: utf-8 -*-
import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import time

print("="*60)
print("DOUBLE BUFFERING PIPELINE")
print("="*60)

model = models.resnet50(weights='DEFAULT').cuda().eval()
MEAN = torch.tensor([0.485,0.456,0.406]).cuda().view(3,1,1)
STD  = torch.tensor([0.229,0.224,0.225]).cuda().view(3,1,1)

N = 200
images = [np.random.randint(0,255,(224,224,3),dtype=np.uint8)
          for _ in range(N)]

# Method 1: Sequential (no overlap)
print("Method 1: Sequential...")
with torch.no_grad():
    for _ in range(20):
        img = torch.from_numpy(images[0]).float().cuda()
        _ = model((img.permute(2,0,1).unsqueeze(0)/255.0-MEAN)/STD)
torch.cuda.synchronize()

lat_seq = []
with torch.no_grad():
    for i in range(N):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        img = torch.from_numpy(images[i]).float().cuda()
        img = (img.permute(2,0,1).unsqueeze(0)/255.0 - MEAN)/STD
        _ = model(img)
        e.record()
        torch.cuda.synchronize()
        lat_seq.append(s.elapsed_time(e))
lat_seq = np.array(lat_seq)

# Method 2: Double buffering (overlap transfer + compute)
print("Method 2: Double buffering...")
stream_compute  = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()

# Pre-allocate two pinned buffers
buf_a = torch.empty(224*224*3, dtype=torch.float32).pin_memory()
buf_b = torch.empty(224*224*3, dtype=torch.float32).pin_memory()

with torch.no_grad():
    for _ in range(20):
        buf_a.copy_(torch.from_numpy(
            images[0].flatten().astype(np.float32)))
        img = buf_a.cuda(non_blocking=True).reshape(224,224,3)
        img = (img.permute(2,0,1).unsqueeze(0)/255.0-MEAN)/STD
        _ = model(img)
torch.cuda.synchronize()

lat_db = []
with torch.no_grad():
    for i in range(N):
        buf = buf_a if i % 2 == 0 else buf_b
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        buf.copy_(torch.from_numpy(
            images[i].flatten().astype(np.float32)))
        with torch.cuda.stream(stream_transfer):
            img = buf.cuda(non_blocking=True)
        stream_compute.wait_stream(stream_transfer)
        with torch.cuda.stream(stream_compute):
            img = img.reshape(224,224,3)
            img = (img.permute(2,0,1).unsqueeze(0)/255.0-MEAN)/STD
            _ = model(img)
        e.record()
        torch.cuda.synchronize()
        lat_db.append(s.elapsed_time(e))
lat_db = np.array(lat_db)

print(f"\n{'='*65}")
print(f"DOUBLE BUFFERING RESULTS")
print(f"{'='*65}")
print(f"{'Metric':<22} {'Sequential':>15} {'Double Buffer':>15} {'Gain':>10}")
print(f"{'-'*65}")
print(f"{'Average (ms)':<22} {lat_seq.mean():>15.2f} {lat_db.mean():>15.2f} {lat_seq.mean()/lat_db.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<22} {lat_seq.std():>15.2f} {lat_db.std():>15.2f}")
print(f"{'99th %ile (ms)':<22} {np.percentile(lat_seq,99):>15.2f} {np.percentile(lat_db,99):>15.2f}")
print(f"{'='*65}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Double Buffering Pipeline\nOverlapping Transfer + Compute',
             fontweight='bold')

axes[0].hist(lat_seq, bins=40, alpha=0.7, color='#e74c3c',
             label=f'Sequential (mean={lat_seq.mean():.1f}ms)')
axes[0].hist(lat_db,  bins=40, alpha=0.7, color='#27ae60',
             label=f'Double Buffer (mean={lat_db.mean():.1f}ms)')
axes[0].set_xlabel('Latency (ms)')
axes[0].set_ylabel('Count')
axes[0].set_title('Latency Distribution')
axes[0].legend()
axes[0].grid(alpha=0.3)

methods = ['Sequential', 'Double\nBuffer']
means   = [lat_seq.mean(), lat_db.mean()]
stds    = [lat_seq.std(),  lat_db.std()]
colors  = ['#e74c3c', '#27ae60']
bars = axes[1].bar(methods, means, color=colors, alpha=0.85,
                   width=0.4, yerr=stds, capsize=5)
axes[1].set_ylabel('Latency (ms)')
axes[1].set_title('Average Latency Comparison')
for bar, m in zip(bars, means):
    axes[1].text(bar.get_x()+bar.get_width()/2, m+0.3,
                 f'{m:.2f}ms', ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('C:/gpu_optimizer/results/double_buffering.png',
            dpi=150, bbox_inches='tight')
print("\nChart saved: double_buffering.png")
plt.show()

np.save('C:/gpu_optimizer/results/double_buffer_latencies.npy', lat_db)
print("Results saved.")
