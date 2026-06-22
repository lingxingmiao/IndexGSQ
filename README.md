# IndexGSQ
1GB嵌入向量使用~90MB内存进行索引，并在2.5bit保持92%+的召回率(383,113个向量)

代码全部由AI完成
等待写完基础方法

### 核心要点 
- 向量重排（先使用Faiss的IVF进行粗排再逐一计算余弦相似度选取相似度最高的k-1个向量，在块内共有k个向量）
- GSQ_K（旋转块缩放方向来降低多个向量同维度下的方差）

核心内容全部取自[TranslatorMinecraft](https://github.com/lingxingmiao/Translator-Minecraft/)的R1.6 B2版本
