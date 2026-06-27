import torch
import torchvision.models as models
import numpy as np
import cv2
import time

print("="*50)
print("CUDA STREAMS: PIPELINE OPTIMIZATION")
print("="*50)

model = models.resnet50(weights='DEFAULT').cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)

def preprocess_cuda(img_np, stream=None):
    with torch.cuda.stream(stream) if stream else torch.no_grad():
        img_tensor = torch.from_numpy(img_np).cuda().float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        img_resized = torch.nn.functional.interpolate(
            img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return (img_resized - MEAN) / STD

# Create test images
N = 100
images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(N)]

# Warmup
with torch.no_grad():
    for i in range(10):
        inp = preprocess_cuda(images[i])
        _ = model(inp)
torch.cuda.synchronize()

# Method 1: Sequential (no streams)
print("Method 1: Sequential (no streams)...")
latencies_seq = []
with torch.no_grad():
    for i in range(N):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        inp = preprocess_cuda(images[i])
        out = model(inp)
        end.record()
        torch.cuda.synchronize()
        latencies_seq.append(start.elapsed_time(end))

# Method 2: With CUDA Streams (overlap)
print("Method 2: CUDA Streams (overlapped)...")
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
latencies_stream = []

with torch.no_grad():
    for i in range(N):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.cuda.stream(stream1):
            inp = preprocess_cuda(images[i], stream1)

        stream2.wait_stream(stream1)
        with torch.cuda.stream(stream2):
            out = model(inp)

        end.record()
        torch.cuda.synchronize()
        latencies_stream.append(start.elapsed_time(end))

latencies_seq    = np.array(latencies_seq)
latencies_stream = np.array(latencies_stream)

print(f"\n{'='*65}")
print(f"{'Metric':<22} {'Sequential':>14} {'CUDA Streams':>14} {'Speedup':>10}")
print(f"{'-'*65}")
print(f"{'Average (ms)':<22} {latencies_seq.mean():>14.2f} {latencies_stream.mean():>14.2f} {latencies_seq.mean()/latencies_stream.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<22} {latencies_seq.std():>14.2f} {latencies_stream.std():>14.2f} {latencies_seq.std()/latencies_stream.std():>9.2f}x")
print(f"{'Min (ms)':<22} {latencies_seq.min():>14.2f} {latencies_stream.min():>14.2f}")
print(f"{'Max (ms)':<22} {latencies_seq.max():>14.2f} {latencies_stream.max():>14.2f}")
print(f"{'99th %ile (ms)':<22} {np.percentile(latencies_seq,99):>14.2f} {np.percentile(latencies_stream,99):>14.2f}")
print(f"{'='*65}")

np.save('C:/gpu_optimizer/results/streams_latencies.npy', latencies_stream)
np.save('C:/gpu_optimizer/results/sequential_latencies.npy', latencies_seq)
print("\nResults saved.")
