# FluxInfer Architecture

FluxInfer is designed to address the "Memory Wall" and "Compute Bound" limitations of modern LLM inference. This document details the internal mechanisms of the engine.

## 1. High-Level Design

The system operates on a **Split-Plane Architecture**:

*   **Control Plane (Python)**: Handles request scheduling, API termination, and high-level routing logic. It communicates with the data plane via PyO3 bindings.
*   **Data Plane (Rust)**: Manages memory, kernel execution graphs, and tensor operations. This layer is "Zero-GIL" (Global Interpreter Lock) compliant, allowing true parallelism.

## 2. Tensor Paging (Memory Management)

Standard KV-caching allocates contiguous memory blocks for each request. This leads to **fragmentation** and **over-provisioning**.

FluxInfer implements **PagedAttention**, inspired by OS virtual memory paging:

1.  **Block Table**: We divide the KV cache into fixed-size blocks (default: 16KB).
2.  **Virtual Mapping**: Each request sees a contiguous "virtual" memory space for its keys and values.
3.  **Physical Allocation**: Blocks are allocated non-contiguously in VRAM as needed.

### Mathematical Model

Let $B$ be the block size. The memory waste $W$ per sequence is bounded by:

$$ W < \frac{B}{S_{total}} $$

Where $S_{total}$ is the total sequence length. By tuning $B$, we achieve near-zero external fragmentation.

## 3. Composable Optimization Graph

FluxInfer treats inference optimizations as a Directed Acyclic Graph (DAG).

### Nodes
- **Source**: Input Token Embeddings
- **Op**: Attention, FFN, Norm
- **Sink**: Logits

### Compiler Passes
When `pipeline.compile()` is called:

1.  **Fusion Pass**: Adjacent element-wise operations (e.g., RMSNorm + Scale) are fused into single CUDA kernels.
2.  **Quantization Pass**: Weight matrices are converted to Int4/Int8 formats if `QuantizationMode` is enabled.
3.  **Routing Pass**: MoE gating layers are injected with complexity-aware dispatch logic.

## 4. Adaptive MoE Routing

For Mixture-of-Experts models, we use a lightweight predictor network $P(x)$ to estimate the computational cost of a query.

$$ \text{Experts}(x) = \text{TopK}(\text{Softmax}(W_g x + \epsilon), k) $$

FluxInfer dynamically adjusts $k$ (number of active experts) based on system load and query difficulty:

- **Low Load**: Use more experts for higher quality.
- **High Load**: Use fewer experts (approximate) to maintain throughput.
