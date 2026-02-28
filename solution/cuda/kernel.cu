/*
 * CUDA Kernel Template for FlashInfer Competition.
 *
 * Implement your kernel logic here. The entry point function name should match
 * the `entry_point` setting in config.toml.
 *
 * See the track definition for required function signature and semantics.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

__global__ void gdn_v1() {
    /*
     * Your CUDA kernel implementation.
     *
     * TODO: Implement your kernel according to the track definition.
     * The function signature should match the track requirements.
     */
}

int main() {
    int B=2;
    int T=1;
    int num_heads=16;
    int K=128;
    int V=128;

    size_t size_q = B*T*num_heads*K*sizeof(half);
    size_t size_k = B*T*num_heads*K*sizeof(half);
}
