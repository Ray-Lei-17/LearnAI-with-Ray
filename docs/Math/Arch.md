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
- **外推性”极差（最致命）：** 因为 BERT 的位置矩阵大小是固定的（比如 512）。如果你训练时最长只见过 512 个词，那么当你突然想让它处理 513 个词时，它会发现矩阵里没有“第 512 行”（索引从 0 开始），模型直接崩溃，完全无法处理。
    
- **无法处理超长文本：** 想要支持更长的文本，就得无限扩大这个矩阵，不仅浪费参数，而且模型没见过的位置它永远学不会。

### ROPE (Rotary Positional Embedding，旋转位置嵌入)
**RoPE (Rotary Positional Embedding，旋转位置嵌入)** 是目前大模型（LLM）界的绝对主流，Llama、PaLM、Qwen、GLM 等几乎所有顶尖模型都在用它。

如果说 BERT 的方法是给每个座位贴**固定标签**，那么 RoPE 的核心思想就是：**“旋转”**。

---

**1. 核心直觉：把向量“转”起来**

在 RoPE 之前，我们要么在向量上**加**一个数字（Transformer），要么给向量**加**一个学出来的特征（BERT）。

RoPE 换了个思路：它把 Embedding 向量看作是在高维空间里的一个个点，然后**根据这个词所在的位置，把这个点旋转一定的角度。**

- **第 1 个词：** 旋转 $\theta$ 度。
    
- **第 2 个词：** 旋转 $2\theta$ 度。
    
- **第 $m$ 个词：** 旋转 $m\theta$ 度。
    

---

**2. 数学实现：从 2D 平面开始**

为了方便理解，我们假设你的 Embedding 向量只有 2 维，即 $(x_1, x_2)$。在数学中，我们可以把这个 2 维向量看作是一个**复数**。

RoPE 的操作就是：给这个复数乘以一个指数因子 $e^{im\theta}$。

根据欧拉公式，这等同于应用一个旋转矩阵：

$$\begin{pmatrix} q'_1 \\ q'_2 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$$

- **$m$** 是词的位置索引。
    
- **$\theta$** 是一个预设的频率常数。
    
- 对于高维向量（比如 768 维），RoPE 会把它们**每两个维度分成一组**，每一组都在各自的 2D 平面上旋转。
    

---

**3. 为什么 RoPE 这么厉害？（三个神级特性）**

**A. 完美的相对位置表达**

这是 RoPE 最天才的地方。虽然我们旋转的是绝对位置（第 $m$ 个词转 $m\theta$），但在计算 Attention 时，Query ($q$) 和 Key ($k$) 之间做点积（Dot Product），奇迹发生了：

点积的结果只跟它们的“相对距离” $(m-n)$ 有关！

> **直观理解：** 就像两个在钟表盘上的人，一个人在 2 点，一个人在 5 点。虽然他们各自有绝对位置，但他们之间的夹角（3 小时的距离）是固定的。Attention 机制只关心这个夹角。

**B. 远程衰减（Long-term Decay）**

RoPE 的数学推导自然带有一个特性：**距离越远的词，它们旋转后的重合度（相关性）倾向于越低。** 这非常符合人类语言习惯——通常离得近的词关系更紧密。

**C. 优秀的外推性（Extrapolation）**

还记得 BERT 遇到第 513 个词会崩溃吗？

RoPE 是靠函数计算出来的，没有固定的“表”。虽然训练时可能只用了 2048 的长度，但如果你想测 4096，RoPE 依然能算出旋转角度。

配合一些技巧（如 NTK-Aware 插值），RoPE 可以让模型在不经过微调的情况下，直接处理比训练时长得多的文本。

#### GPT解释
我个人觉得GPT解释更全面，因为ROPE除了计算方法上的差别，还有一个很重要的差别是其应用的位置，不是直接把位置向量加到输入上，而是直接把 **Query / Key 向量按位置做旋转**的，这一点很重要，但是太长了我先不放在这里，欢迎查看[[ROPE]]。

### 总结对比

| **特性**   | **传统 Transformer (Sin)** | **BERT (Learned)** | **RoPE (Rotary)**          |
| -------- | ------------------------ | ------------------ | -------------------------- |
| **操作方式** | 相加 (Add)                 | 相加 (Add)           | **乘法/旋转 (Multiplicative)** |
| **位置类型** | 绝对位置                     | 绝对位置               | **绝对形式，相对本质**              |
| **外推性**  | 一般                       | 极差 (固定长度)          | **极好 (支持长度扩展)**            |
| **当前地位** | 早期基石                     | 判别式模型常用            | **生成式大模型标配**               |


## KV Cache
在我们前面说Attention的时候，QKV都需要由输入X乘以一个矩阵来得到，在顺序生成时，前面的token的KV在这个期间是不会发生改变的，所以预存下来是最划算的，但是这在长序列时就带来很多的空间开销

### Multi-Query Attention(MQA)
![[Pasted image 20260115103035.png]]
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
主要的出发点也是为了节省KV Cache，这里相当于把两个KV合成压缩成一个小的矩阵，就是图中蓝色的Compressed的部分，然后存下来，要使用的时候，乘以各自的矩阵来重建成key和value，同时压缩的矩阵维度也会更小，这样增加了矩阵乘法的次数，但是可以减少内存使用。

#### Mixture-of-Experts (MoE)
![[Pasted image 20260115104201.png]]
MoE 的核心思想是用多个专家层替换 Transformer 块中的每个Feed forward模块，其中每个专家层也是一个Feed forward模块。这意味着我们用一个Feed forward块替换了多个Feed forward块。

由于一次只有少数专家处于活跃状态，MoE 模块通常被称为稀疏的，这与始终使用完整参数集的密集模块形成对比。然而，通过 MoE 增加的参数总数提高了 LLM 的容量，这意味着它在训练过程中可以获取更多知识。稀疏性保持了推理的高效性，因为我们并非同时使用所有参数。

例如，DeepSeek-V3 每个 MoE 模块有 256 个专家，总共有 671B参数。但在推理过程中，一次只有 9 个专家处于活跃状态（1 个共享专家加上 8 个由路由器选择的专家）。这意味着每次推理步骤只使用 37B参数，而不是全部 671B。

DeepSeek-V3 的 MoE 设计的一个显著特点是使用了共享专家。这是一种对所有标记始终活跃的专家。这个想法并不新颖，在 DeepSeek 2024 MoE 和 2022 DeepSpeedMoE 论文中已经介绍过。

共享专家的优势最早在 DeepSpeedMoE 论文中被提出，他们发现与没有共享专家相比，这能提升整体建模性能。这可能是由于常见或重复的模式不需要由多个独立专家学习，从而让他们有更多空间去学习更专业的模式。

#### Feed Forward (SwiGLU)
这个地方的feed forward也进行了一些发展和变化，最传统的Feed Forward就是纯粹的全连接层，这个地方的SwiGLU是加了一个门控机制GLU，然后使用激活函数SiLU(Swish)：`silu(x) = x * sigmoid(x)`

![[Pasted image 20260115112032.png]]
![[Pasted image 20260115112047.png]]
![[Pasted image 20260115112108.png]]
![[Pasted image 20260115112157.png]]
![[Pasted image 20260115112227.png]]
![[Pasted image 20260115112248.png]]
![[Pasted image 20260115112305.png]]

### OLMo 2
![[Pasted image 20260115112654.png]]
> Figure: A comparison of Post-Norm, Pre-Norm, and OLMo 2's flavor of Post-Norm.
#### Normalization Layer Placement
Norm放置的位置有几个说法，可以放在注意力/前馈模块之后，称为 Post-LN 或 Post-Norm。之前传统的transformer的Post-Norm位置有些微的差别，是放在residual内部的。

2020 年，Xiong 等人表明，Pre-LN在初始化时能产生更稳定的梯度。此外，研究人员提到，预 LN 甚至在不进行仔细的学习率预热的情况下也能表现良好，而学习率预热通常是Post-LN的关键工具。

移动Post-LN能有效提高训练稳定性，但图中展示的并不是单独的贡献，还有QK-Norm。
![[Pasted image 20260115114949.png]]

#### QK-Norm
QK 范数本质上又是另一个 RMSNorm 层。它被放置在多头注意力（MHA）模块内部，并在应用 RoPE 之前应用于查询（q）和键（k）。

如前所述，QK-Norm 与 Post-Norm 一起稳定了训练。需要注意的是，QK-Norm 并非由 OLMo 2 发明，而是追溯到 2023 年关于扩展视觉 Transformer 的论文。

### Gemma 3

#### Sliding Window Attention

滑动窗口注意力相当于把注意力从全局限制到一个局部视野，能够有效减少KV Cache。

![[Pasted image 20260116094642.png]]

那么，什么是滑动窗口注意力？如果我们把常规自注意力视为全局注意力机制，因为每个序列元素都可以访问其他所有序列元素，那么我们可以把滑动窗口注意力视为局部注意力，因为我们在这里限制了当前查询位置周围的上下文大小。下面的图说明了这一点。

![[Pasted image 20260116095034.png]]

如上所述，滑动窗口注意力也被称为局部注意力，因为局部窗口会围绕并随着当前查询位置移动。相比之下，常规注意力是全局的，因为每个标记都可以访问所有其他标记。Gemma 2 使用了一种混合注意力机制，该机制以 1:1 的比例结合滑动窗口（局部）和全局注意力。每个标记可以关注附近上下文中的 4k 标记窗口。而 Gemma 3 现在采用了 5:1 的比例，这意味着每 5 个滑动窗口（局部）注意力层中只有 1 个是完整的注意力层；此外，滑动窗口的大小从 4096（Gemma 2）减少到了 1024（Gemma 3）。这使模型更加注重高效、局部的计算。

#### Normalization Layer Placement in Gemma 3
![[Pasted image 20260116095436.png]]
Gemma3既有Pre-Norm又有Post-Norm

### Mistral Small 3.1

![[Pasted image 20260116095654.png]]

Mistral Small 3.1 比 Gemma 3 具有更低推理延迟的原因可能在于其自定义的分词器，以及减小了 KV 缓存和层数。

### Llama 4

![[Pasted image 20260116101700.png]]

首先，Llama 4 使用与前序工作类似的Grouped-Query Attention，而 DeepSeek-V3 使用我们在文章开头讨论过的Multi-Head Latent Attention。Llama 4 Maverick 采用更经典的 MoE 设置，与 DeepSeek-V3 相比，其专家数量更少但规模更大（每个激活专家的隐藏大小为 8,192，而 DeepSeek-V3 每个激活专家的隐藏大小为 2,048）。此外，DeepSeek 在每个 transformer 块中使用 MoE 层（前 3 个除外），而 Llama 4 则在交替的 transformer 块中使用 MoE 层和密集模块。

### Qwen3

0.6B 模型可能是目前最小的开源权重模型。根据我的个人经验，考虑到其体积小，它的表现非常出色。如果你打算在本地运行，它具有很高的每秒令牌吞吐量和低内存占用。更重要的是，由于体积小，它在本地（用于教育目的）也易于训练。

#### Qwen3 (Dense)

![[Pasted image 20260116102247.png]]

#### 6.2 Qwen3 (MoE) 

密集模型通常在微调、部署和跨各种硬件优化方面更为直接。另一方面，MoE 模型针对推理扩展进行了优化。例如，在固定的推理预算下，它们可以在不按比例增加推理成本的情况下，实现更高的整体模型容量（即由于模型更大而在训练过程中吸收更多知识）。总结这一部分，让我们看看 Qwen3 235B-A22B（注意 A22B 代表“22B 个激活参数）到 DeepSeek-V3，后者具有近两倍的激活参数（37B）。

![[Pasted image 20260116102617.png]]

如上图所示，DeepSeek-V3 和 Qwen3 235B-A22B 架构非常相似。值得注意的是，Qwen3 模型已经不再使用共享专家（早期的 Qwen 模型，如 Qwen2.5-MoE 曾使用共享专家）。

更新。Qwen3 的开发者之一林宇阳回应如下：

>当时我们没有发现共享专家有足够显著的改进，并且我们担心共享专家对推理优化的影响。这个问题老实说没有直接的答案。

### SmolLM3

![[Pasted image 20260116102811.png]]

最有趣的一点是其使用 NoPE（No Positional Embeddings）

#### No Positional Embeddings (NoPE)

在 LLM 的语境中，NoPE 是一个较旧的概念，追溯到 2023 年的一篇论文([The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466))，目的是移除显式的位置信息注入（例如通过早期 GPT 架构中的经典绝对位置嵌入层或如今的 RoPE）。

在基于 transformer 的 LLMs 中，位置编码通常是必要的，因为自注意力机制将 token 视为独立于顺序的。绝对位置嵌入通过添加一个额外的嵌入层来解决这个问题，该层向 token 嵌入中添加信息。另一方面，RoPE 通过相对于其标记位置旋转查询和键向量来解决这一问题。

然而，在 NoPE 层中，**根本不添加任何这样的位置信号**：既不是固定的，也不是学习的，更不是相对的。什么都没有。

尽管没有位置嵌入，但由于**因果注意力掩码**的存在，模型仍然知道哪些标记在前。这种掩码阻止每个标记关注未来的标记。因此，位置 t 的标记只能看到位置≤t 的标记，从而保持了自回归顺序。所以，虽然模型结构中没有显式添加位置信息，但仍然隐含着方向性，LLM 在基于梯度下降的常规训练中，如果发现利用这种方向性有利于优化目标，可以学会利用它。（想了解更多信息，可以查看 NoPE 论文的定理。）

总的来说，NoPE 论文不仅发现没有必要注入位置信息，还发现 NoPE 具有更好的长度泛化能力，这意味着随着序列长度的增加，LLM 的答题性能下降得较少，如图所示。

![[Pasted image 20260116103428.png]]

请注意，上述实验是在一个相对较小的、大约有 1 亿参数的 GPT 风格模型和相对较小的上下文大小下进行的。不清楚这些发现是否适用于更大、更现代的 LLMs。因此，SmolLM3 团队很可能只在每第 4 层“应用”了 NoPE（或者更准确地说，省略了 RoPE）。

### Kimi K2 and Kimi K2 Thinking

一个值得注意的方面是它使用了相对较新的 Muon 优化器的一个变体，而不是 AdamW。据我所知，这是 Muon 首次用于此类规模的任何生产模型（之前它仅展示可扩展到 16B）。这导致了非常漂亮的训练损失曲线，这很可能帮助该模型跃居上述基准测试的榜首。模型本身有 1 万亿个参数，这确实令人印象深刻。

![[Pasted image 20260116103800.png]]

如上图所示，Kimi K2 基本上与 DeepSeek V3 相同，只是它在 MoE 模块中使用了更多的专家，而在 Multi-head Latent Attention（MLA）模块中使用了更少的头。

### GPT-OSS

OpenAI 发布了 gpt-oss-120b 和 gpt-oss-20b，这是自 2019 年 GPT-2 以来 OpenAI 首次发布的开源权重模型。

![[Pasted image 20260116104233.png]]

该架构包含了我们在之前讨论的其他架构中见过的所有熟悉组件。将较小的 gpt-oss 架构与 Qwen3 30B-A3B 并置，后者也是一个 MoE 模型，具有相似数量的活跃参数（gpt-oss 有 3.6B 活跃参数，而 Qwen3 30B-A3B 有 3.3B）。gpt-oss 使用滑动窗口注意力机制（类似于 Gemma 3，但在每隔一层而不是使用 5:1 的比例中使用）。

#### Width Versus Depth

gpt-oss 和 Qwen3 使用相似的组件。但如果我们仔细观察这两个模型，我们会发现 Qwen3 的架构更深，拥有 48 个 transformer 块而不是 24 个。另一方面，gpt-oss 的架构更宽：

- 一个 2880 的嵌入维度而不是 2048
- 一个中间专家（前馈）投影维度也是 2880 而不是 768

也值得注意的一点是，gpt-oss 使用了双倍数量的注意力头，但这并不直接增加模型的宽度。宽度是由嵌入维度决定的。

在参数数量固定的情况下，哪种方法更具优势？一般来说，更深层的模型具有更高的灵活性，但由于不稳定性问题（如梯度爆炸和梯度消失，RMSNorm 和快捷连接旨在缓解这些问题），训练起来可能更困难。

更宽的架构在推理时具有优势（具有更高的每秒 token 吞吐量），这是由于更好的并行化，但代价是更高的内存成本。

在建模性能方面，不幸的是，除了 Gemma 2 论文中的消融研究（表 9）之外，我并没有意识到有任何好的、直接的可比性研究（保持参数大小和数据集不变），该研究发现，对于 9B 参数的架构，更宽的设置比更深层的设置略好。在 4 个基准测试中，更宽的模型平均得分为 52.0，而更深层的模型平均得分为 50.8。

#### Few Large Versus Many Small Experts

如上图 27 所示，值得注意的是，gpt-oss 的专家数量出奇地少（32 个而不是 128 个），并且每个 token 只使用 4 个而不是 8 个激活专家。然而，每个专家都比 Qwen3 中的专家大得多。

这很有趣，因为最近的趋势和发展表明，更多、更小的模型是有益的。在总参数量不变的情况下，这种变化在 DeepSeekMoE 论文下面的图 28 中得到了很好地说明。

![[Pasted image 20260116105314.png]]

#### Attention Bias and Attention Sinks

gpt-oss 和 Qwen3 都使用分组查询注意力机制。主要区别在于，如前文所述，gpt-oss 通过在每个第二层使用滑动窗口注意力来限制上下文大小。然而，有一个有趣的细节引起了我的注意。似乎 gpt-oss 在注意力权重中使用偏置单元，如下面的图 29 所示。

 ![[Pasted image 20260116105641.png]]

自从 GPT-2 时代以来，我没有见过这些偏差单元的使用，它们通常被认为是冗余的。确实，我发现一篇最近的论文从数学上证明了这一点至少适用于关键转换（ `k_proj` ）。此外，实证结果表明，有无偏差单元之间的差异很小。

![[Pasted image 20260116105708.png]]

你可能注意到的另一个细节是图 30 中代码截图里对汇流点的定义。在一般模型中，注意力汇流点是放置在序列开头的特殊“始终被关注”的标记，用于稳定注意力，这在长上下文场景中尤其有用。也就是说，如果上下文变得非常长，序列开头的这个特殊被关注的标记仍然会被关注，并且可以学习存储一些关于整个序列的通用信息。（我认为这个想法最初是在《带有注意力汇流点的有效流式语言模型》论文中提出的。）

在 gpt-oss 实现中，注意力汇流点并不是输入序列中的实际标记。相反，它们是附加到注意力分数上的学习到的每个head中的偏差逻辑值。其目标与前面提到的注意力汇流点相同，但不会修改标记化的输入。

### Grok 2.5

xAI 发布了他们 2700 亿参数的 Grok 2.5 模型的权重。从架构上看，Grok 2.5 整体上看起来相当标准（图 xx），但有几个值得注意的细节。

![[Pasted image 20260116110021.png]]

例如，Grok 2.5 使用少量大型专家（八个），这反映了较旧的趋势。如前所述，较新的设计（如 DeepSeekMoE 论文中的设计）倾向于使用更多的小型专家（Qwen3 中也存在这种设计）。另一个有趣的选择是使用所谓的共享专家。图中左侧所示的增加的 SwiGLU 模块充当始终开启的共享专家。它与经典共享专家设计并不完全相同，因为它的中间维度被加倍了，但思想是相同的。（作者仍然觉得 Qwen3 省略了共享专家这一点很有趣，而且很有趣的是，Qwen4 及以后的模型是否会改变这一点。）

### GLM-4.5

![[Pasted image 20260116110341.png]]

GLM-4.5是一种指令/推理混合型，类似于 Qwen3，但在函数调用和代理式上下文中进行了更好的优化。GLM-4.5 有两种变体。旗舰级的 3550 亿参数模型在 12 项基准测试中平均优于 Claude 4 Opus，仅略逊于 OpenAI 的 o3 和 xAI 的 Grok 4。此外还有 GLM-4.5-Air，这是一个更紧凑的 1060 亿参数版本，性能仅略低于 3550 亿参数模型。

设计大体相似，但 GLM-4.5 采用了 DeepSeek V3 首次引入的结构选择：3 个密集层位于专家混合（MoE）模块之前。为什么？从多个密集层开始可以提高大型 MoE 系统的收敛稳定性和整体性能。如果立即引入 MoE 路由，稀疏专家选择的不可稳定性会干扰早期的句法和语义特征提取。因此，可以说通过保持初始层为密集层，确保模型在路由决策开始塑造高级处理之前形成稳定的低级表示。

此外，GLM-4.5 使用与 DeepSeek-V3 类似（与 Qwen3 不同）的共享专家，也保留了 GPT-2 和 gpt-oss 中使用的注意力偏差机制。

### Qwen3-Next

2025 年 9 月 11 日，Qwen3 团队发布了 Qwen3 Next 80B-A3B，提供指令和思考两种变体。

#### Expert Size and Number

新的 Qwen3 Next 架构脱颖而出，因为它比之前的 235B-A22B 模型小 3 倍，但引入了四倍的专家数量，甚至增加了共享专家。

![[Pasted image 20260116110923.png]]

#### Gated DeltaNet + Gated Attention Hybrid

另一个亮点是，他们用门控 DeltaNet + 门控注意力混合机制取代了常规的注意力机制，这有助于实现内存使用方面的原生 262k token 上下文长度（之前的 235B-A22B 模型原生支持 32k，通过 YaRN 扩展支持 131k）。

那么这个新的注意力混合机制是如何工作的呢？与分组查询注意力（GQA）相比——后者仍然是标准的缩放点积注意力（如前所述，通过跨查询头组共享 K/V 来减少 KV 缓存大小和内存带宽，但其解码成本和缓存仍随序列长度增长），他们的混合机制以图 36 所示 3:1 的比例混合门控 DeltaNet 模块和门控注意力模块。

![[Pasted image 20260116111153.png]]
> Gated DeltaNet + Gated Attention 混合机制。注意这些是以 3:1 的比例排列的，意味着 3 个带有 Gated DeltaNet 的 Transformer 块后跟 1 个带有 Gated Attention 的 Transformer 块。

我们可以将门控注意力模块视为标准缩放点积注意力，它可用于 GQA，但它在上面做了一些调整。门控注意力和普通 GQA 模块之间的主要区别是：

1. 一个输出门（由 Sigmoid 控制，通常按通道进行）在注意力结果被加回到残差之前对其进行缩放；
2. QKNorm 使用零中心的 RMSNorm，而不是标准的 RMSNorm；
3. 部分 RoPE（在部分维度上）。

请注意，这些本质上只是 GQA 的稳定性变化。

门控 DeltaNet 是一个更显著的变化。在 DeltaNet 模块中，q、k、v 和两个门（α、β）由带有归一化的线性层和轻量级卷积层产生，该层用快速权重 [_delta rule_](https://arxiv.org/abs/2412.06464)更新替换了注意力机制。然而，权衡在于 DeltaNet 提供的基于内容的检索不如全注意力精确，这也是为什么仍然保留了一个门控注意力层。

鉴于注意力机制的计算复杂度呈平方级增长，DeltaNet 组件被加入以帮助提高内存效率。在“线性时间、无缓存”系列中，DeltaNet 模块本质上是一个 Mamba 的替代方案。Mamba 维护一个带有学习状态空间滤波器的状态（本质上是对时间进行动态卷积）。DeltaNet 维护一个由α和β更新的微小快速权重内存，并用 q 读取它，仅使用小的卷积来帮助形成 q、k、v、α、β。

#### Multi-Token Prediction

Qwen3 还在顶部增加了一种效率技术： [Multi-Token Prediction](https://arxiv.org/abs/2404.19737)（MTP）。（请注意，DeepSeek V3 & V3.2，以及后来的 GLM-4.5 和 MiniMax-M2 在训练中都使用 MTP；然而，由于这是一种训练技术，作者在架构比较中并没有明确讨论它。）

多 token 预测训练 LLM 在每一步预测多个未来 token，而不是一个。在这里，在每个位置 t，小的额外头部（线性层）输出 t+1…t+k 的 logits，并且为这些偏移量（在 MTP 论文中，研究人员建议 k=4）求交叉熵损失。这个附加信号加快了训练，推理可能仍然保持一次生成一个 token。然而，额外的头部可以用于推测性多 token 解码，这似乎是 Qwen3-Next 所做的，但是细节仍然有点稀疏：

> Qwen3-Next 引入了一种原生的多标记预测（MTP）机制，这不仅产生了一个对推测解码具有高接受率的 MTP 模块，还提升了整体性能。此外，Qwen3-Next 针对 MTP 的多步推理性能进行了专门优化，通过保持训练与推理之间的一致性，通过多步训练进一步提高了推测解码在实际场景中的接受率。来源：[Qwen3-Next blog post](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)

### MiniMax-M2

[MiniMax-M1](https://arxiv.org/abs/2506.13585)使用了一种线性注意力变体（lightning attention），这种注意力机制比常规（full）注意力更高效。如下图所示，作者将 MiniMax-M2 与其他解码器风格的 Transformer LLMs 分组，因为它没有使用 MiniMax-M1 中提出的有效闪电注意力变体。相反，开发者们回到了使用完整注意力，这可能是为了提高建模（和基准）性能。

![[Pasted image 20260116112555.png]]

> 主要 LLMs 的时间线，旁边是一些构成更高效替代方案的关注混合模型，它们在建模性能上有所牺牲，以换取更高的效率。

#### Per-Layer QK-Norm

这里值得注意的一个亮点可能是，MiniMax-M2 使用了一种所谓的 `per_layer` QK-Norm，而不是常规的 QK-Norm。仔细查看代码可以发现，它在注意力机制中是这样实现的：

```
self.q_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_heads, eps=...)

self.k_norm = MiniMaxText01RMSNormTP(self.head_dim * self.total_num_kv_heads, eps=...)
```

在这里， `hidden_size` 等于拼接的头部（ `num_heads * head_dim` ），因此 RMSNorm 具有一个具有每个头部（以及每个头部维度）的不同参数的缩放向量。

所以， `per_layer` 表示 RMSNorm（如前文所述，用于 QK-Norm）在每个 Transformer 模块中定义（与常规 QK-Norm 相同），但除此之外，它不是跨注意力头复用，而是每个注意力头都有一个独特的 QK-Norm。（为什么不叫per-head？）

![[Pasted image 20260116115234.png]]

#### Partial RoPE

最后，与 MiniMax-M1 类似，MiniMax-M2 在注意力模块内部使用“部分”RoPE 而不是常规 RoPE 来编码位置信息。与常规 RoPE 类似，旋转是在应用 QK-Norm 之后应用于查询和键的。

这里的 Partial RoPE 表示每个头的 `rotary_dim` 通道获得旋转位置编码，而其余的 `head_dim - rotary_dim` 通道保持不变。

在官方的 M1 README 文件中，开发者提到

> 旋转位置嵌入（RoPE）应用于注意力头维度的一半，基础频率为 10,000,000

我们可以这样想象：

```
Full RoPE:     [r r r r r r r r]
Partial RoPE:  [r r r r — — — —]
```

在上述概念图示中，"r"表示旋转（位置编码）的维度，虚线表示未受影响的维度。

这样做有什么意义？在 [M1 论文](https://arxiv.org/abs/2501.08313)中，开发者指出

> …在 softmax 注意力维度的一半上实现 RoPE 能够实现长度外推而不降低性能。

作者的推测是，这可以防止长序列“过度”旋转，尤其是那些比训练数据集中最长文档还要长的序列。也就是说，这里的理由可能是没有旋转比模型在训练中未曾见过的“糟糕”或“过于极端”的旋转要好。

### Kimi Linear （TODO）

### Olmo 3 Thinking （TODO）

### DeepSeek V3.2 （TODO）

![[Pasted image 20260116115709.png]]

### Mistral 3 Large （TODO）

### Nemotron 3 （TODO）

![[Pasted image 20260116115854.png]]

Nemotron 3 Nano（30B-A3B）是一个 52 层的混合 Mamba-Transformer 模型，它将 Mamba-2 序列建模模块与稀疏专家混合（MoE）前馈层交错排列，并且只在少数层中使用自注意力机制。