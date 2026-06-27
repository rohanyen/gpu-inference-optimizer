import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time

print("="*50)
print("RAW CUDA KERNEL BENCHMARK")
print("="*50)

kernel_code = """
__global__ void preprocess_kernel(
    const unsigned char* input,
    float* output,
    int in_h, int in_w,
    int out_h, int out_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_w || out_y >= out_h) return;

    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;
    float in_x = out_x * scale_x;
    float in_y = out_y * scale_y;

    int x0 = (int)in_x;
    int y0 = (int)in_y;
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);

    float dx = in_x - x0;
    float dy = in_y - y0;

    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; c++) {
        float p00 = input[(y0 * in_w + x0) * 3 + c];
        float p01 = input[(y0 * in_w + x1) * 3 + c];
        float p10 = input[(y1 * in_w + x0) * 3 + c];
        float p11 = input[(y1 * in_w + x1) * 3 + c];

        float val = p00*(1-dx)*(1-dy) + p01*dx*(1-dy)
                  + p10*(1-dx)*dy     + p11*dx*dy;

        val = ((val / 255.0f) - mean[c]) / std[c];
        output[c * out_h * out_w + out_y * out_w + out_x] = val;
    }
}
"""

print("Compiling raw CUDA kernel...")
mod = SourceModule(kernel_code)
preprocess_fn = mod.get_function("preprocess_kernel")
print("Kernel compiled!")

in_h, in_w = 640, 640
out_h, out_w = 224, 224

img_np = np.random.randint(0, 255, (in_h, in_w, 3), dtype=np.uint8)
img_gpu = cuda.mem_alloc(img_np.nbytes)
cuda.memcpy_htod(img_gpu, img_np)

out_gpu = cuda.mem_alloc(3 * out_h * out_w * 4)

block = (16, 16, 1)
grid = ((out_w + 15)//16, (out_h + 15)//16, 1)

# Warmup
for _ in range(20):
    preprocess_fn(img_gpu, out_gpu,
        np.int32(in_h), np.int32(in_w),
        np.int32(out_h), np.int32(out_w),
        block=block, grid=grid)
cuda.Context.synchronize()

# Benchmark raw kernel
N = 1000
start = time.perf_counter()
for _ in range(N):
    preprocess_fn(img_gpu, out_gpu,
        np.int32(in_h), np.int32(in_w),
        np.int32(out_h), np.int32(out_w),
        block=block, grid=grid)
cuda.Context.synchronize()
kernel_time = (time.perf_counter() - start) / N * 1000

# Benchmark OpenCV
img_cpu = np.random.randint(0, 255, (in_h, in_w, 3), dtype=np.uint8)
start = time.perf_counter()
for _ in range(N):
    r = cv2.resize(img_cpu, (out_w, out_h))
    n = (r.astype(np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    _ = n.transpose(2,0,1)
opencv_time = (time.perf_counter() - start) / N * 1000

print(f"\n{'='*60}")
print(f"RAW CUDA KERNEL vs OpenCV CPU")
print(f"{'='*60}")
print(f"{'Method':<30} {'Latency':>10} {'Throughput':>15}")
print(f"{'-'*60}")
print(f"{'Raw CUDA Kernel (.cu)':<30} {kernel_time:>9.3f}ms {1000/kernel_time:>14.0f} img/s")
print(f"{'PyTorch CUDA':<30} {'0.595':>9}ms {'1679':>14} img/s")
print(f"{'OpenCV (CPU)':<30} {opencv_time:>9.3f}ms {1000/opencv_time:>14.0f} img/s")
print(f"{'-'*60}")
print(f"{'Raw CUDA vs OpenCV':<30} {opencv_time/kernel_time:>9.2f}x")
print(f"{'='*60}")

np.save('C:/gpu_optimizer/results/raw_kernel_time.npy', np.array([kernel_time]))
print("\nResults saved.")
