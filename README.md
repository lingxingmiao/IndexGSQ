# IndexGSQ
Indexing 1GB of embedding vectors requires only 75–90MB of memory, while maintaining a 92%+ recall rate at 2.5 bits (383,113 vectors).

### Recall Rates
- 8-bit: up to 100%
- 6-bit: up to 99.9%
- 4-bit: up to 99.5%
- 3-bit: up to 98.4%
- 2-bit: up to 93.9%

All code is entirely AI-generated.

### Core Concepts
- **Vector Reranking**: First, coarse ranking is performed using Faiss's IVF, followed by one-by-one cosine similarity computation to select the top $k-1$ most similar vectors. Together with the anchor, there are a total of $k$ vectors within the block.
- **GSQ_K**: Rotates block scaling directions to reduce variance across multiple vectors within the same dimension.

### Major Drawbacks
- Unable to dynamically add vectors (due to the combined effects of vector reranking and GSQ_K).
- High memory consumption during vector import.

### Indexing Methods
- **Speed Ranking**: `IndexGSQKCosineMoE` > `IndexGSQKCosineFast` > `IndexGSQKCosine`
- **Memory Ranking**: `IndexGSQKCosine` < `IndexGSQKCosineFast` < `IndexGSQKCosineMoE`
- **Recall ranking**: `IndexGSQKCosine` = `IndexGSQKCosineFast` > `IndexGSQKCosineMoE`

`IndexGSQKCosine` and `IndexGSQKCosineFast` are benchmarked against `faiss.IndexFlatIP`.

`IndexGSQKCosineMoE` is benchmarked against `faiss.IndexIVFScalarQuantizer`.

Core content and recall rates are entirely sourced from version R1.6 B2 of [TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/).

---

## Installation & Usage

### Installation
1. Copy the `indexgsq.py` file to your project directory.
2. Install the required dependencies: 
   ```bash
   pip install numpy numba faiss-cpu
   ```

### Usage
```python
import indexgsq
import numpy as np
import time

np.random.seed(42) # Fixed seed for testing

number_of_vectors = 100_000
dimension = 1024
vectors = np.random.randn(number_of_vectors, dimension).astype(np.float32) # Assume these are your vectors
query = np.random.randn(10, dimension).astype(np.float32) # 10 query vectors

# ==========================================
# 1. Basic Exhaustive Search (Benchmarked against faiss.IndexFlatIP)
# ==========================================
# If your project uses Faiss, simply replace faiss.IndexFlatIP with IndexGSQKCosineFast
# vectors_block   : Quantization sub-block size (affects memory and quantization precision)
# reranker_block  : Reranking clustering block size (affects reranking speed and local continuity)
# reranker_factor : Coarse search multiplier during reranking (larger means better reranking quality, but slower build time)
# quantization    : Quantization bits, options: 2, 3, 4, 6, 8

index_fast = indexgsq.IndexGSQKCosineFast(
    vectors_block=128, 
    reranker_block=128, 
    reranker_factor=4, 
    quantization=2
)

print("Building Fast index...")
start_time = time.time()
index_fast.add(vectors) # Includes reranking and quantization, time-consuming, peak memory is about 2-3x the original data
print(f"Index build time: {time.time() - start_time:.2f} seconds")

start_time = time.time()
scores, ids = index_fast.search(query, k=10) # Search Top-10
print(f"Fast search time: {time.time() - start_time:.4f} seconds")
print("Fast Top-10 IDs (first Query):", ids[0])

# Map back to original texts
texts = [f"Text_{i}" for i in range(number_of_vectors)]
top_texts = [texts[i] for i in ids[0] if i != -1]
print("Fast Top-10 Texts:", top_texts)


# ==========================================
# 2. MoE Ultra-Fast Routing Search (Benchmarked against faiss.IndexIVF)
# ==========================================
# Suitable for ultra-large-scale datasets, achieves asymmetric ultra-fast search via SVD rotation and Lloyd-Max routing
index_moe = indexgsq.IndexGSQKCosineMoE(
    vectors_block=128,
    reranker_block=128,
    quantization=2,
    moe_exp=np.float32(0.1) # Activate 10% of expert blocks (similar to IVF's nprobe ratio). uint32 specifies the exact number of experts to activate, float32 specifies the activation ratio.
)

print("\nBuilding MoE index...")
start_time = time.time()
index_moe.add(vectors) # MoE includes the process of training routing features, first-time build is slower
print(f"MoE index build time: {time.time() - start_time:.2f} seconds")

start_time = time.time()
# nprobe controls the number of blocks activated per query. Smaller is faster, larger is more accurate. Alternatively, you can use index_moe.exp=5
scores_moe, ids_moe = index_moe.search(query, k=10, nprobe=5) 
print(f"MoE search time: {time.time() - start_time:.4f} seconds")
print("MoE Top-10 IDs (first Query):", ids_moe[0])


# ==========================================
# 3. Serialization & Persistence
# ==========================================
# Save index (automatically saves Config and routing features)
indexgsq.write_index(index_fast, "index_fast.gsqk")
indexgsq.write_index(index_moe, "index_moe.gsqk")

# Load index (automatically recognizes mode and restores config, no need to pass parameters again)
loaded_index = indexgsq.read_index("index_fast.gsqk")
```

---

### Repository Files
- `indexgsq.py`: Standard version, intended for use as a library in your projects.
- `TranslatorIndexGSQ.py`: Special version tailored for [TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/), designed to be compatible with [TranslatorLib.py](https://github.com/lingxingmiao/Translator-Minecraft/blob/main/TranslatorLib.py) and non-standard input methods. The algorithm is identical to `indexgsq.py`.
