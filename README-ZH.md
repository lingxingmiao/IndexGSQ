# IndexGSQ
1GB嵌入向量使用75-90MB内存进行索引，并在2.5bit保持92%+的召回率(383,113个向量)

### 召回率
- 8位最高100%
- 6位最高99.9%
- 4位最高99.5%
- 3位最高98.4%
- 2位最高93.9%

代码全部由AI完成

### 核心要点 
- 向量重排（先使用Faiss的IVF进行粗排再逐一计算余弦相似度选取相似度最高的k-1个向量，在块内共有k个向量）
- GSQ_K（旋转块缩放方向来降低多个向量同维度下的方差）
- 
### 核心缺点
- 无法动态添加向量（向量重排+GSQ_K所影响）
- 导入向量时需要占用大量内存

### 索引方法
- 速度排名: IndexGSQKCosineMoE > IndexGSQKCosineFast > IndexGSQKCosine
- 内存排名: IndexGSQKCosine < IndexGSQKCosineFast < IndexGSQKCosineMoE

IndexGSQKCosine 与 IndexGSQKCosineFast 对标 faiss.IndexFlatIP

IndexGSQKCosineMoE 对标 faiss.IndexIVFScalarQuantizer



部分核心内容、召回率 全部取自[TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/)的R1.6 B2版本

## 安装与使用
### 安装
1. 复制indexgsq.py文件到您的项目下
2. 安装需要的依赖 `pip install numpy numba faiss-cpu`

### 使用
```python
## 安装与使用
### 安装
1. 复制 `indexgsq.py` 文件到您的项目下
2. 安装需要的依赖 `pip install numpy numba faiss-cpu`

### 使用
```python
import indexgsq
import numpy as np
import time

np.random.seed(42) # 测试使用固定种子

number_of_vectors = 100_000
dimension = 1024
vectors = np.random.randn(number_of_vectors, dimension).astype(np.float32) # 假设这是你的向量
query = np.random.randn(10, dimension).astype(np.float32) # 10个查询向量

# ==========================================
# 1. 基础全量检索 (对标 faiss.IndexFlatIP)
# ==========================================
# 如果你的项目使用的是 Faiss，仅需将 faiss.IndexFlatIP 替换为 IndexGSQKCosineFast 即可
# vectors_block   : 量化子块大小 (影响内存和量化精度)
# reranker_block  : 重排聚类块大小 (影响重排速度和局部连续性)
# reranker_factor : 重排时的粗搜乘数 (越大重排质量越好，但构建越慢)
# quantization    : 量化位数，可选 2, 3, 4, 6, 8

index_fast = indexgsq.IndexGSQKCosineFast(
    vectors_block=128, 
    reranker_block=128, 
    reranker_factor=4, 
    quantization=2
)

print("开始构建 Fast 索引...")
start_time = time.time()
index_fast.add(vectors) # 包含重排与量化，耗时较长，内存峰值约为原数据的2-3倍
print(f"索引构建耗时：{time.time() - start_time:.2f}秒")

start_time = time.time()
scores, ids = index_fast.search(query, k=10) # 搜索 Top-10
print(f"Fast 搜索耗时：{time.time() - start_time:.4f}秒")
print("Fast Top-10 索引 (第一个Query)：", ids[0])

# 映射回原始文本
texts = [f"文本_{i}" for i in range(number_of_vectors)]
top_texts = [texts[i] for i in ids[0] if i != -1]
print("Fast Top-10 文本：", top_texts)


# ==========================================
# 2. MoE 极速路由检索 (对标 faiss.IndexIVF)
# ==========================================
# 适合超大规模数据集，通过 SVD 旋转和 Lloyd-Max 路由实现非对称极速检索
index_moe = indexgsq.IndexGSQKCosineMoE(
    vectors_block=128,
    reranker_block=128,
    quantization=2,
    moe_exp=np.float32(0.1) # 激活 10% 的专家块 (类似 IVF 的 nprobe 比例) uint32为激活专家数量 float32为比例激活
)

print("\n开始构建 MoE 索引...")
start_time = time.time()
index_moe.add(vectors) # MoE 包含训练路由特征的过程，首次构建较慢
print(f"MoE 索引构建耗时：{time.time() - start_time:.2f}秒")



start_time = time.time()
# nprobe 控制每次查询激活的块数量，越小越快，越大越准 或者也可以使用index_moe.exp=5
scores_moe, ids_moe = index_moe.search(query, k=10, nprobe=5) 
print(f"MoE 搜索耗时：{time.time() - start_time:.4f}秒")
print("MoE Top-10 索引 (第一个Query)：", ids_moe[0])


# ==========================================
# 3. 序列化与持久化
# ==========================================
# 保存索引 (自动保存 Config 与路由特征)
indexgsq.write_index(index_fast, "index_fast.gsqk")
indexgsq.write_index(index_moe, "index_moe.gsqk")

# 加载索引 (自动识别模式并恢复配置，无需重新传入参数)
loaded_index = indexgsq.read_index("index_fast.gsqk")
```

### 仓库文件
- `indexgsq.py` : 常规版本, 作为项目的库使用
- `TranslatorIndexGSQ.py` : [TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/)特供版本, 用于兼容[TranslatorLib.py](https://github.com/lingxingmiao/Translator-Minecraft/blob/main/TranslatorLib.py)与非常规传入方式. 算法与`indexgsq.py`相同
