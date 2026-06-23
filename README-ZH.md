# IndexGSQ
1GB嵌入向量使用~90MB内存进行索引，并在2.5bit保持92%+的召回率(383,113个向量)

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
### 核心缺点
- 无法动态添加向量（向量重排+GSQ_K所影响）
- 文本需要同步重排
- 导入向量时需要占用大量内存
- 索引速度较慢（我的电脑是 2.5个/s）

核心内容、召回率 全部取自[TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/)的R1.6 B2版本

## 安装与使用
### 安装
1. 复制indexgsq.py文件到您的项目下
2. 安装需要的依赖 `pip install numpy numba faiss-cpu tqdm`

### 使用
```python
import indexgsq
import numpy as np
import time

np.random.seed(42) # 测试使用固定种子

index = indexgsq.IndexGSQKCosine(vectors_block=128, reranker_block=128, quantization=2) # 创建索引（括号内为默认参数 可不写）
#index = indexgsq.IndexGSQKCosineFast(vectors_block=128, reranker_block=128, quantization=2) # 高速方法 需要消耗更多内存
# vectors_block  : 每个子块包含的向量数（用于量化）
# reranker_block : 重排时每个聚类块的大小
# quantization   : 量化位数，可选 2,3,4,6,8

number_of_vectors = 100_000
vectors = np.random.randn(number_of_vectors, 1024).astype(np.float32) # 假设这是你的向量 这里共39.0625MB
index.add(vectors) # 添加向量到索引（该步骤耗时较长 内存占用较高 为添加向量的二倍）

query = np.random.randn(100, 1024).astype(np.float32) # 假设这是你要查询的向量
start_time = time.time()
scores, ids = index.search(query, k=10) # 搜索前10个
print(f"搜索耗时：{time.time() - start_time:.4f}秒")
print("Top-10索引：", ids[0])

texts = [f"文本_{i}" for i in range(number_of_vectors)] # 假设这是你要索引的文本
reordered_texts = index.text(texts) # 重新排序（必须）
top_texts = [reordered_texts[i] for i in ids[0] if i != -1]
print("Top-10文本：", top_texts)

index.save_text(texts) # 导入文本（可选 自动重排）
index.save("index.gsqk") # 保存索引（后缀不限）

new_index = indexgsq.load("index.gsqk") # 加载索引
new_texts = new_index.texts # 上面导入的文本
```
