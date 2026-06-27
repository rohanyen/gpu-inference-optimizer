# -*- coding: utf-8 -*-
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

print("="*60)
print("CUDA KERNEL EVOLUTION: v1 -> v5")
print("="*60)

# V1: Global memory (Phase 1)
kernel_v1 = """
__global__ void preprocess_v1(
    const unsigned char* input, float* output,
    int in_h, int in_w, int out_h, int out_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= out_w || out_y >= out_h) return;
    float scale_x=(float)in_w/out_w, scale_y=(float)in_h/out_h;
    float in_x=out_x*scale_x, in_y=out_y*scale_y;
    int x0=min((int)in_x,in_w-2), y0=min((int)in_y,in_h-2);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy=in_y-y0;
    float mean[3]={0.485f,0.456f,0.406f};
    float std_v[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){
        float p00=input[(y0*in_w+x0)*3+c];
        float p01=input[(y0*in_w+x1)*3+c];
        float p10=input[(y1*in_w+x0)*3+c];
        float p11=input[(y1*in_w+x1)*3+c];
        float val=p00*(1-dx)*(1-dy)+p01*dx*(1-dy)
                 +p10*(1-dx)*dy+p11*dx*dy;
        output[c*out_h*out_w+out_y*out_w+out_x]=
            ((val/255.0f)-mean[c])/std_v[c];
    }
}
"""

# V2: Shared memory tiling (Phase 2)
kernel_v2 = """
#define BLOCK_W 16
#define BLOCK_H 16
#define TILE_W 32
#define TILE_H 32
__global__ void preprocess_v2(
    const unsigned char* input, float* output,
    int in_h, int in_w, int out_h, int out_w)
{
    __shared__ float tile_r[TILE_H][TILE_W];
    __shared__ float tile_g[TILE_H][TILE_W];
    __shared__ float tile_b[TILE_H][TILE_W];
    int out_x=blockIdx.x*BLOCK_W+threadIdx.x;
    int out_y=blockIdx.y*BLOCK_H+threadIdx.y;
    float scale_x=(float)in_w/out_w, scale_y=(float)in_h/out_h;
    int in_base_x=(int)(blockIdx.x*BLOCK_W*scale_x);
    int in_base_y=(int)(blockIdx.y*BLOCK_H*scale_y);
    for(int dy=0;dy<2;dy++){
        for(int dx=0;dx<2;dx++){
            int ty=threadIdx.y*2+dy, tx=threadIdx.x*2+dx;
            if(ty<TILE_H && tx<TILE_W){
                int gy=min(in_base_y+ty,in_h-1);
                int gx=min(in_base_x+tx,in_w-1);
                int idx=(gy*in_w+gx)*3;
                tile_r[ty][tx]=input[idx+0];
                tile_g[ty][tx]=input[idx+1];
                tile_b[ty][tx]=input[idx+2];
            }
        }
    }
    __syncthreads();
    if(out_x>=out_w||out_y>=out_h) return;
    float in_x=out_x*scale_x-in_base_x;
    float in_y=out_y*scale_y-in_base_y;
    int x0=min((int)in_x,TILE_W-2), y0=min((int)in_y,TILE_H-2);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy2=in_y-y0;
    float mean[3]={0.485f,0.456f,0.406f};
    float std_v[3]={0.229f,0.224f,0.225f};
    float r=tile_r[y0][x0]*(1-dx)*(1-dy2)+tile_r[y0][x1]*dx*(1-dy2)
           +tile_r[y1][x0]*(1-dx)*dy2+tile_r[y1][x1]*dx*dy2;
    float g=tile_g[y0][x0]*(1-dx)*(1-dy2)+tile_g[y0][x1]*dx*(1-dy2)
           +tile_g[y1][x0]*(1-dx)*dy2+tile_g[y1][x1]*dx*dy2;
    float b=tile_b[y0][x0]*(1-dx)*(1-dy2)+tile_b[y0][x1]*dx*(1-dy2)
           +tile_b[y1][x0]*(1-dx)*dy2+tile_b[y1][x1]*dx*dy2;
    output[0*out_h*out_w+out_y*out_w+out_x]=((r/255.0f)-mean[0])/std_v[0];
    output[1*out_h*out_w+out_y*out_w+out_x]=((g/255.0f)-mean[1])/std_v[1];
    output[2*out_h*out_w+out_y*out_w+out_x]=((b/255.0f)-mean[2])/std_v[2];
}
"""

# V3: Vectorized memory access (float4)
kernel_v3 = """
__constant__ float MEAN[3] = {0.485f, 0.456f, 0.406f};
__constant__ float STD[3]  = {0.229f, 0.224f, 0.225f};

__global__ void preprocess_v3(
    const unsigned char* input, float* output,
    int in_h, int in_w, int out_h, int out_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= out_w || out_y >= out_h) return;

    float scale_x=(float)in_w/out_w, scale_y=(float)in_h/out_h;
    float in_x=out_x*scale_x, in_y=out_y*scale_y;
    int x0=min((int)in_x,in_w-2), y0=min((int)in_y,in_h-2);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy=in_y-y0;

    // Use constant memory for mean/std (cached)
    for(int c=0;c<3;c++){
        float p00=input[(y0*in_w+x0)*3+c];
        float p01=input[(y0*in_w+x1)*3+c];
        float p10=input[(y1*in_w+x0)*3+c];
        float p11=input[(y1*in_w+x1)*3+c];
        float val=p00*(1-dx)*(1-dy)+p01*dx*(1-dy)
                 +p10*(1-dx)*dy+p11*dx*dy;
        // Use constant memory (faster than registers for broadcast)
        output[c*out_h*out_w+out_y*out_w+out_x]=
            ((val/255.0f)-MEAN[c])/STD[c];
    }
}
"""

# V4: Warp-level optimization + loop unrolling
kernel_v4 = """
__constant__ float MEAN4[3] = {0.485f, 0.456f, 0.406f};
__constant__ float STD4[3]  = {0.229f, 0.224f, 0.225f};

__global__ void preprocess_v4(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w, int out_h, int out_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= out_w || out_y >= out_h) return;

    float scale_x=(float)in_w/out_w, scale_y=(float)in_h/out_h;
    float in_x=out_x*scale_x, in_y=out_y*scale_y;
    int x0=min((int)in_x,in_w-2), y0=min((int)in_y,in_h-2);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy=in_y-y0;
    float w00=(1-dx)*(1-dy), w01=dx*(1-dy);
    float w10=(1-dx)*dy,     w11=dx*dy;

    // Manually unrolled loop (no branch overhead)
    int base00=(y0*in_w+x0)*3, base01=(y0*in_w+x1)*3;
    int base10=(y1*in_w+x0)*3, base11=(y1*in_w+x1)*3;

    // Channel 0
    float v0=input[base00]*w00+input[base01]*w01+
             input[base10]*w10+input[base11]*w11;
    output[0*out_h*out_w+out_y*out_w+out_x]=((v0/255.0f)-MEAN4[0])/STD4[0];

    // Channel 1
    float v1=input[base00+1]*w00+input[base01+1]*w01+
             input[base10+1]*w10+input[base11+1]*w11;
    output[1*out_h*out_w+out_y*out_w+out_x]=((v1/255.0f)-MEAN4[1])/STD4[1];

    // Channel 2
    float v2=input[base00+2]*w00+input[base01+2]*w01+
             input[base10+2]*w10+input[base11+2]*w11;
    output[2*out_h*out_w+out_y*out_w+out_x]=((v2/255.0f)-MEAN4[2])/STD4[2];
}
"""

# V5: All optimizations combined (best)
kernel_v5 = """
#define BLOCK_W 16
#define BLOCK_H 16
#define TILE_W 32
#define TILE_H 32

__constant__ float MEAN5[3] = {0.485f, 0.456f, 0.406f};
__constant__ float STD5[3]  = {0.229f, 0.224f, 0.225f};

__global__ void preprocess_v5(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w, int out_h, int out_w)
{
    __shared__ unsigned char smem[TILE_H][TILE_W][3];

    int out_x=blockIdx.x*BLOCK_W+threadIdx.x;
    int out_y=blockIdx.y*BLOCK_H+threadIdx.y;
    float scale_x=(float)in_w/out_w, scale_y=(float)in_h/out_h;
    int in_base_x=(int)(blockIdx.x*BLOCK_W*scale_x);
    int in_base_y=(int)(blockIdx.y*BLOCK_H*scale_y);

    // Cooperative load into shared memory
    int tid=threadIdx.y*BLOCK_W+threadIdx.x;
    for(int i=tid; i<TILE_H*TILE_W; i+=BLOCK_W*BLOCK_H){
        int ty=i/TILE_W, tx=i%TILE_W;
        int gy=min(in_base_y+ty,in_h-1);
        int gx=min(in_base_x+tx,in_w-1);
        smem[ty][tx][0]=input[(gy*in_w+gx)*3+0];
        smem[ty][tx][1]=input[(gy*in_w+gx)*3+1];
        smem[ty][tx][2]=input[(gy*in_w+gx)*3+2];
    }
    __syncthreads();

    if(out_x>=out_w||out_y>=out_h) return;

    float in_x=out_x*scale_x-in_base_x;
    float in_y=out_y*scale_y-in_base_y;
    int x0=min((int)in_x,TILE_W-2), y0=min((int)in_y,TILE_H-2);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy2=in_y-y0;
    float w00=(1-dx)*(1-dy2), w01=dx*(1-dy2);
    float w10=(1-dx)*dy2,     w11=dx*dy2;

    // Unrolled channels + shared memory + constant memory
    float v0=smem[y0][x0][0]*w00+smem[y0][x1][0]*w01+
             smem[y1][x0][0]*w10+smem[y1][x1][0]*w11;
    float v1=smem[y0][x0][1]*w00+smem[y0][x1][1]*w01+
             smem[y1][x0][1]*w10+smem[y1][x1][1]*w11;
    float v2=smem[y0][x0][2]*w00+smem[y0][x1][2]*w01+
             smem[y1][x0][2]*w10+smem[y1][x1][2]*w11;

    output[0*out_h*out_w+out_y*out_w+out_x]=((v0/255.0f)-MEAN5[0])/STD5[0];
    output[1*out_h*out_w+out_y*out_w+out_x]=((v1/255.0f)-MEAN5[1])/STD5[1];
    output[2*out_h*out_w+out_y*out_w+out_x]=((v2/255.0f)-MEAN5[2])/STD5[2];
}
"""

# Compile all kernels
print("Compiling all 5 kernels...")
kernels = {
    'V1 Global Memory':     (SourceModule(kernel_v1).get_function("preprocess_v1"), {}),
    'V2 Shared Memory':     (SourceModule(kernel_v2).get_function("preprocess_v2"), {}),
    'V3 Constant Memory':   (SourceModule(kernel_v3).get_function("preprocess_v3"), {}),
    'V4 Loop Unrolled':     (SourceModule(kernel_v4).get_function("preprocess_v4"), {}),
    'V5 All Combined':      (SourceModule(kernel_v5).get_function("preprocess_v5"), {}),
}
print("All compiled!")

in_h, in_w = 640, 640
out_h, out_w = 224, 224
img_np = np.random.randint(0,255,(in_h,in_w,3),dtype=np.uint8)
img_g  = cuda.mem_alloc(img_np.nbytes)
out_g  = cuda.mem_alloc(3*out_h*out_w*4)
cuda.memcpy_htod(img_g, img_np)

block = (16,16,1)
grid  = ((out_w+15)//16,(out_h+15)//16,1)
N = 2000

# OpenCV baseline
img_cpu = np.random.randint(0,255,(in_h,in_w,3),dtype=np.uint8)
t0 = time.perf_counter()
for _ in range(N):
    r = cv2.resize(img_cpu,(out_w,out_h))
    n = (r.astype(np.float32)/255.0-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    _ = n.transpose(2,0,1)
cv_time = (time.perf_counter()-t0)/N*1000

results = {}
print(f"\n{'Kernel':<22} {'Time(ms)':>10} {'vs OpenCV':>12} {'vs V1':>10}")
print("-"*58)

v1_time = None
for name, (fn, _) in kernels.items():
    # Warmup
    for _ in range(50):
        fn(img_g, out_g,
           np.int32(in_h), np.int32(in_w),
           np.int32(out_h), np.int32(out_w),
           block=block, grid=grid)
    cuda.Context.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        fn(img_g, out_g,
           np.int32(in_h), np.int32(in_w),
           np.int32(out_h), np.int32(out_w),
           block=block, grid=grid)
    cuda.Context.synchronize()
    t = (time.perf_counter()-t0)/N*1000

    if v1_time is None: v1_time = t
    results[name] = t
    print(f"{name:<22} {t:>10.3f} {cv_time/t:>11.2f}x {v1_time/t:>9.2f}x")

print(f"\n{'OpenCV CPU':<22} {cv_time:>10.3f} {'1.00x':>12}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('CUDA Kernel Evolution: V1 to V5\n640x640 -> 224x224 Preprocessing',
             fontweight='bold')

names  = list(results.keys()) + ['OpenCV\nCPU']
times  = list(results.values()) + [cv_time]
colors = ['#3498db','#2ecc71','#e67e22','#9b59b6','#e74c3c','#95a5a6']

bars = ax.bar(names, times, color=colors, alpha=0.85, width=0.6)
ax.set_ylabel('Latency (ms)')
ax.set_title('Preprocessing Latency Comparison')
for bar, t in zip(bars, times):
    ax.text(bar.get_x()+bar.get_width()/2, t+0.001,
            f'{t:.3f}ms', ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('C:/gpu_optimizer/results/kernel_evolution.png',
            dpi=150, bbox_inches='tight')
print("\nChart saved: kernel_evolution.png")
plt.show()

np.save('C:/gpu_optimizer/results/kernel_times.npy',
        np.array(list(results.values())))
print("Results saved.")
