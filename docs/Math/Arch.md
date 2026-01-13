## Architecture

### Attention
#### Standard scaled dot-product attention
![[Pasted image 20251219103049.png]]
#### Multi-head
![[Pasted image 20251219103147.png]]
![[Pasted image 20251219103155.png]]
![[Pasted image 20251219103207.png]]
![[Pasted image 20251219103216.png]]

#### Bahdanau (additive) attention
![[Pasted image 20251219103301.png]]

#### Luong attention (general)
![[Pasted image 20251219103330.png]]


### Multi-Head Softmax Attention
总体过程描述：输入X，与不同的QKV参数矩阵相乘得到QKV，计算Attention，将不同的head合并 起来再过一个output的矩阵
![[Pasted image 20260104163233.png]]
![[Pasted image 20260104163302.png]]


#### 存在的问题
##### Massive Activation
在训练和推理大模型时，研究人员发现隐藏层（Hidden States）中极少数维度的数值会突然变得巨大（比中位数大 10,000 倍以上）。这种现象被称为 **Massive Activation**。

- **特征：**
    
    - **极少数维度：** 在几千维的向量中，可能只有 1-2 个维度会出现这种巨值。
        
    - **特定位置：** 它们通常出现在模型的**中间层**，并且往往集中在**起始 Token（如 `<bos>`）**或标点符号（如换行符、句号）上。
        
    - **输入无关：** 无论你输入什么内容，这些特定的维度总是会“爆表”。
        
- **影响：**
    
    - **量化困难：** 这是模型量化（从 FP16 转为 INT8/INT4）最大的敌人。因为动态范围太广，如果为了照顾这些巨值，普通数值的精度就会损失殆尽。
        
    - **隐含偏置（Implicit Bias）：** 研究发现这些巨值实际上充当了模型的“偏置项”（Bias），即使它们对应的词没有实际意义，模型也需要它们来维持计算的稳定性。

##### Attention Sink
**Attention Sink** 指的是模型倾向于将大量的注意力权重（Attention Score）分配给序列中的**前几个 Token（通常是第一个 Token）**，即使这些词在语义上完全无关。

- **为什么会产生？**
    
    - **Softmax 的强迫性：** Attention 层使用 Softmax 函数，要求所有权重之和必须为 1。
        
    - **无处安放的注意力：** 当模型在当前序列中找不到与当前预测相关的上下文时，它必须把“剩下的注意力”放到某个地方。
        
    - **首位优势：** 由于起始 Token（如 `<s>` 或第一个词）在自回归训练中对所有后续词都可见，它就像一个“垃圾桶”或“汇聚槽”，承接了那些多余的、无意义的注意力分数。
        
- **应用：StreamingLLM**
    
    - 过去模型处理超长文本时，如果删掉开头的词，模型会瞬间崩溃（PPL 飙升）。
        
    - **StreamingLLM** 发现：只要在滑动窗口中**永远保留前 4 个 Token**（即 Attention Sink），模型就可以在不进行微调的情况下，处理几十万甚至无限长的文本。

最近的研究（如 2024 年的论文）指出，Massive Activation 和Attention Sink两个现象其实是**“同一枚硬币的两面”**：

1. **因果关系：** 因为第一个 Token（Attention Sink）在计算过程中产生了 **Massive Activation**（极大的模长），导致它在计算注意力点积时，更容易获得极高的分数。
    
2. **功能一致：** Massive Activation 提供了数值上的“背景偏置”，而 Attention Sink 提供了结构上的“背景锚点”。
    
3. **防止过拟合：** 这种机制让模型在处理不相关的上下文时，能够通过将注意力“倾倒”给第一个 Token，从而避免强行学习无关词之间的虚假关联。


### RMSNorm
**RMSNorm**（Root Mean Square Layer Normalization，均方根层归一化）是目前大语言模型（如 **Llama**、**Gopher**、**PaLM**）中几乎标配的归一化方法。

它是对传统 **LayerNorm** 的一种简化和优化。研究发现，LayerNorm 带来的性能提升主要来自“重缩放”（Re-scaling），而“重移位”（Re-centering，即减去均值）并不是必须的。

---

#### 1. 核心公式对比

为了理解 RMSNorm，我们先看看它和普通 LayerNorm 的区别：

##### LayerNorm (标准层归一化)

LayerNorm 做两件事：

1. **平移**：减去均值 $\mu$，使分布中心为 0。
    
2. **缩放**：除以标准差 $\sigma$，使方差为 1。
    

$$y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} \cdot \gamma + \beta$$

##### RMSNorm (均方根归一化)

RMSNorm 认为减去均值是不必要的。它直接根据神经元输出的**均方根**进行缩放：

$$y = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{其中} \quad \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2 + \epsilon}$$

---

#### 2. 为什么要用 RMSNorm？

1. **计算更高效**：
    
    - LayerNorm 需要计算均值和方差。
        
    - RMSNorm 只需要计算平方和的平均值。
        
    - 在硬件实现上，RMSNorm 减少了计算步骤（少了减法），计算开销降低了约 **10%~40%**。
        
2. **数值稳定性**：
    
    - RMSNorm 依然能够保持激活值的尺度（Scale）稳定，防止梯度爆炸或消失，这对于训练上千亿参数的模型至关重要。
        
3. **性能不减**：
    
    - 实验证明，去掉“平移”操作（即不减均值）并不会降低模型的准确率。既然效果一样，当然选更快的。
        

---

#### 3. 在 Transformer 里的位置变化（Pre-Norm）

在早期的 Transformer（如 BERT）中，归一化通常放在残差连接之后（Post-Norm）。

而在现在的 LLM（如 Llama）中，RMSNorm 被放在了 Attention 或 MLP 层之前（Pre-Norm），并且不再使用偏置项 $\beta$。

这种“Pre-RMSNorm”结构让深层模型的训练变得更加稳定。

---

#### 总结：两者的直观区别

|**特性**|**LayerNorm**|**RMSNorm**|
|---|---|---|
|**操作**|减均值 + 除以标准差|**仅**除以均方根|
|**可学习参数**|$\gamma$ (增益), $\beta$ (偏置)|$\gamma$ (增益)|
|**计算复杂度**|较高|**较低 (更省算力)**|
|**主流模型**|BERT, GPT-2|**Llama (1/2/3), Mistral, DeepSeek**|

**通俗地说：** LayerNorm 是把数据强行拉到坐标原点再缩放，而 RMSNorm 只是把数据的长度缩放到一个标准范围，但不移动中心点。


### Gated Attention

来自阿里的文章，在attention的计算中加入gate，效果最佳是在SDPA之后加入
![[Pasted image 20260104121004.png]]
![[Pasted image 20260104121019.png]]
在Attention中引入gate其中有效的关键之一在引入了non-linearity
![[Pasted image 20260104121114.png]]
其次之一是引入了Input-Dependent Sparsity，从实验来看，得到的gating score是很多为0，非常稀疏，所以作者认为可能**gating score sparsity may filter out irrelevant contextual information for the query**。
对于Gate的形式，会有一些变种，但是效果最佳的是这个默认设置。**Unless otherwise specified, we employ head-specific, multiplicative gating utilizing the sigmoid activation function**。
![[Pasted image 20260104122154.png]]在加入了gate之后，attention sink和massive activation得到情况也消失了。


## Position Embedding

### 绝对位置编码（Absolute Positional Encoding）

代表模型：Vanilla Transformer

具体方法是使用**正余弦周期函数（Sinusoidal Position Encoding）**

##### 数学公式

对于一个位置 $pos$ 和维度索引 $i$，其位置编码 $PE$ 的计算方式如下：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- **$pos$**：词在句子中的绝对位置（0, 1, 2...）。
    
- **$i$**：向量维度的索引。
    
- **$d_{model}$**：模型的嵌入维度（比如 512）。

#### 为什么要用正余弦函数？

设计者选择这种特殊的函数形式，主要基于以下几个深层考虑：

- **唯一性与周期性：** 每个位置的编码向量在 $d_{model}$ 维空间中是唯一的，且不同频率的波形组合形成了一种类似于“二进制计数器”的模式。
    
- 线性变换特性（最重要的数学特性）：
    
    由于三角函数的感法公式，对于任何固定的偏移量 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。这意味着模型理论上可以很容易地学习到词与词之间的相对位置关系。
    
- 长度外推性（Extrapolation）：
    
    理论上，正余弦函数是连续的。即使在推理时遇到的句子比训练时更长，模型也能通过函数计算出对应位置的编码（尽管实际效果受限）。
    
- 无需学习参数：
    
    这种编码是固定计算出来的，不增加模型的参数量，能减少训练负担。

### 可学习位置嵌入(Learned Positional Embedding)
简单来说就是保存一个位置嵌入的查找表，每次做嵌入就按照位置去表里找对应的嵌入，加上以后送入训练
#### 形象化解释

**第一步：初始化一个“位置编码矩阵”**

在模型刚建立时，我们会定义一个随机初始化的矩阵。

- **维度：** `[max_seq_len, hidden_size]`
    
- **例子：** 在 BERT-base 中，最大长度（`max_seq_len`）通常是 **512**，向量维度（`hidden_size`）是 **768**。
    
- 那么这个矩阵就像一张有 512 行、768 列的表格。每一行代表一个位置（第 0 行、第 1 行……直到第 511 行）的特征向量。
    

**第二步：给每个 Token 分配位置索引**

当一句话进入 BERT 时，系统会自动给每个词标上它的序号。

- **输入：** “我 爱 机器 学习”
    
- **索引：** `[0, 1, 2, 3]`
    
- 即使是一模一样的词，比如“我 喜欢 我”，第一个“我”的索引是 0，第二个“我”的索引是 2。
    

 **第三步：查表（Lookup）并相加**

就像查字典一样，模型根据索引去“位置编码矩阵”里找到对应的向量。

1. 第 0 个位置的词，就去取矩阵的第 0 行向量。
    
2. 第 1 个位置的词，就去取矩阵的第 1 行向量。
    
3. **关键点：** 将这个**位置向量**直接和**词向量（Token Embedding）**相加。
    

> **公式表示：** $Final\_Embedding = Token\_Embedding + Position\_Embedding$

**第四步：通过训练进行更新**

这是它和传统方法最大的不同：

- 在训练（Pre-training）过程中，模型会根据损失函数（Loss）产生的梯度，通过**反向传播**来不断调整这个矩阵里的数值。
    
- 模型最终会自己学会：**“第 1 个位置”和“第 2 个位置”之间应该有什么样的数学联系。**

#### 缺点


## KV Cache
在我们前面说Attention的时候，QKV都需要由输入X乘以一个矩阵来得到，在顺序生成时，前面的token的KV在这个期间是不会发生改变的，所以预存下来是最划算的，但是这在长序列时就带来很多的空间开销

### Multi-Query Attention(MQA)
如我们前面所说attention有多个head，表示从不同角度提问，但是按照之前的设计，每个不同head都有不同的KV矩阵，这样一来显存的使用量就是
$$
显存 \propto num\_heads \times seq\_len \times dim
$$
head越多，显存增长的越快。而MQA认为不同的head需要不同的Q，但未必需要不同的KV，故将KV在head间共享，从而使显存使用量降低
$$
显存 \propto 1 \times seq\_len \times dim
$$
假设有32个head，则显存直接减少32倍。公式由$Attention_h​(Q_h​,K_h​,V_h​)$变成$Attention_h​(Q_h​,K​,V)$
代价上来说，表达能力会部分下降，但不会特别大。

### Grouped-Query Attention(GQA)
如上面所说，MQA虽然能优化显存使用，但是一定程度会影响性能，其中的折中方案就是GQA，某几个head形成一个组，来共享KV Cache，得到性能和显存的平衡点。比如
- Q：32 heads
- K/V：比如 4 组（每 8 个 Q 共享一组）

此时
$$
显存 \propto num\_groups \times seq\_len \times dim
$$
$num\_groups=4$。

## Transformer Block

### Vanilla Transformer
![[Pasted image 20260104164843.png]]
![[Pasted image 20260104164244.png]]


> 以下内容参考自一个**非常好**的博客：
> [The Big LLM Architecture Comparison](https://sebastianraschka.com/blog/2025/the-big-llm-architecture-comparison.html)
> 只是我提取出其中的部分内容帮助关注重点
### Brief Comparison Figure
![[Pasted image 20260106105544.png]]
对比最开始的decoder block，位置嵌入已经从绝对位置编码发展到旋转位置编码(RoPE)，多头注意力机制基本被分组查询注意力取代，激活函数从GELU发展成更高效的SwiGLU。
### DeepSeek V3/R1
![[Pasted image 20260107151656.png]]
这个地方因为是第一个模型，我们可以对比一下和原来的transformer的decoder的block的区别：
- Norm位置发生改变从计算module后面移到了前面
- 从传统的BatchNorm/LayerNorm变成RNSNorm
- 位置编码绝对位置编码

#### Multi-Head Latent Attention (MLA)
主要的出发点也是为了节省KV Cache，这里相当于把两个KV合成压缩成一个小的矩阵，就是图中蓝色的Compressed的部分，