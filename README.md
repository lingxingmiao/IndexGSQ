# IndexGSQ
1GB of embedding vectors can be indexed using just 75–90MB of memory, while maintaining a 92%+ recall rate at 2.5 bits (383,113 vectors).

### Recall Rates
- 8-bit: up to 100%
- 6-bit: up to 99.9%
- 4-bit: up to 99.5%
- 3-bit: up to 98.4%
- 2-bit: up to 93.9%

All code is entirely AI-generated.

### Core Concepts
- **Vector Reranking** – First, coarse ranking is performed using Faiss's IVF, then cosine similarity is computed one by one to select the top k-1 most similar vectors. Together with the anchor, there are k vectors within the block.
- **GSQ_K** – Rotates and scales block directions to reduce variance across multiple vectors within the same dimension.

### Major Drawbacks
- Unable to dynamically add vectors (due to the combined effects of vector reranking and GSQ_K).
- Texts must be reordered synchronously with the vectors.
- High memory consumption during vector import.
- Relatively slow indexing speed (on my machine: ~12.28 vectors/second for `IndexGSQKCosineFast`).

Core content and recall data are sourced from [TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/) version R1.6 B2.

---

## Installation & Usage

### Installation
1. Copy the `indexgsq.py` file to your project directory.
2. Install the required dependencies:
   ```bash
   pip install numpy numba faiss-cpu tqdm
   ```

### Usage
```python
import indexgsq
import numpy as np
import time

# Use a fixed seed for testing
np.random.seed(42)

# If your project uses Faiss and you want to migrate to IndexGSQ, simply replace 
# faiss.IndexIP (or other algorithms) with indexgsq.IndexGSQKCosineFast or 
# indexgsq.IndexGSQKCosine, then adjust the initialization parameters accordingly.
index = indexgsq.IndexGSQKCosine(vectors_block=128, reranker_block=128, quantization=2)  # Create index (default values shown, can be omitted)
# index = indexgsq.IndexGSQKCosineFast(vectors_block=128, reranker_block=128, quantization=2)  # Fast method, consumes more memory

# vectors_block  : Number of vectors in each sub-block (used for quantization)
# reranker_block : Size of each clustering block during reranking
# quantization   : Quantization bits, options: 2, 3, 4, 6, 8

number_of_vectors = 100_000
vectors = np.random.randn(number_of_vectors, 1024).astype(np.float32)  # Example vectors, ~39.0625MB here
index.add(vectors)  # Add vectors to the index (time-consuming and memory-heavy, roughly 2x the input size)

query = np.random.randn(100, 1024).astype(np.float32)  # Example query vectors
start_time = time.time()
scores, ids = index.search(query, k=10)  # Search top 10
print(f"Search time: {time.time() - start_time:.4f} seconds")
print("Top-10 IDs:", ids[0])

texts = [f"Text_{i}" for i in range(number_of_vectors)]  # Example texts to index
top_texts = [texts[i] for i in ids[0] if i != -1]
print("Top-10 Texts:", top_texts)

indexgsq.write_index(index, "index.gsqk")  # Save the index

new_index = indexgsq.read_index("index.gsqk")  # Load the index
```

---

### Repository Files
- `indexgsq.py` – Standard version, intended for use as a library in your projects.
- `TranslatorIndexGSQ` – Special version tailored for [TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/), designed to be compatible with `TranslatorLib.py` and non-standard input methods. The algorithm is identical to `indexgsq.py`.
