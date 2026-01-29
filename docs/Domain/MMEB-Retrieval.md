以下都是MMEB榜单上的sota

## Qwen3-VL-Embedding-8B

[Qwen3-VL-Embedding & Qwen3-VL-Reranker：统一多模态表征与排序](https://qwen.ai/blog?id=qwen3-vl-embedding)

比较特别的一点是引入了Reranker
- **High-Precision Reranking (Reranker)**: We also introduce the Qwen3-VL-Reranker series to complement the embedding model. The reranker takes a (query, document) pair as input—where both query and document may contain arbitrary single or mixed modalities—and outputs a precise relevance score. In retrieval pipelines, the two models are typically used in tandem: the embedding model performs efficient initial recall, while the reranker refines results in a subsequent re-ranking stage. This two-stage approach significantly boosts retrieval accuracy.  
    高精度重排序（重排序器）：我们还引入了 Qwen3-VL-Reranker 系列来补充嵌入模型。重排序器以（查询，文档）对作为输入——其中查询和文档可能包含任意单一或混合模态——并输出精确的相关性分数。在检索流程中，这两个模型通常协同使用：嵌入模型执行高效的初始召回，而重排序器在后续的重排序阶段优化结果。这种两阶段方法显著提升了检索准确率。

### 模型架构

![[Pasted image 20260123122216.png]]

Qwen3-VL-Embedding 采用**双塔架构**，Qwen3-VL-Reranker采用**单塔架构**。

**Embedding模型**接收单模态或混合模态输入，并将其映射为高维语义向量。具体而言，我们提取基座模型最后一层中对应 `[EOS]` token 的隐藏状态向量，作为输入的最终语义表示。这种方法确保了大规模检索所需的高效独立编码能力。

**Reranking模型**接收输入对 `(Query, Document)` 并进行联合编码。它利用基座模型内的交叉注意力（Cross-Attention）机制，实现 Query 和 Document 之间更深层、更细粒度的跨模态交互和信息融合。模型最终通过预测两个特殊 token（`yes` 和 `no`）的生成概率来表达输入对的相关性分数。

使用的模型都是qwen3-vl

![[Pasted image 20260123122939.png]]

重排方法 重排模型采用点式排序法，根据指令中提供的相关性定义来评估一对多模态实例之间的相关性。输入格式遵循 Qwen3-VL 上下文结构，其中相关性定义指令和待评估的一对多模态实例都作为用户信息传递。这些多模态输入可以是文本、图像、视频或这些模态的任意组合。最后，通过计算模型预测下一个输出标记为 "是 "或 "否 "的概率，获得这对标记的相关性估计。

![[Pasted image 20260123123020.png]]

### 训练数据

![[Pasted image 20260123131539.png]]

#### Positive Refinement and Hard Negative Mining

![[Pasted image 20260123145810.png]]


### 训练策略
![[Pasted image 20260123131740.png]]
不断筛选，训练，再筛选，再训练，再筛选。


## InterestFM-TTE Embedding 

[Think Then Embed: Generative Context Improves Multimodal Embedding](https://arxiv.org/pdf/2510.05014)
### Method

![[Pasted image 20260123150907.png]]

作者观察到，对于需要指令理解的检索任务，现有的基于编码器的方法在需要复杂推理和上下文基础的复杂指令上往往表现不佳。为此，我们引入了一个明确的“思考”阶段，通过利用思维链（Chain-of-Thought）推理轨迹，在生成嵌入之前进行思考。思考阶段由一个从 7B 到 70B 的 LLM 推理器处理，为 7B 嵌入器提供有用的上下文标记。在不增加额外数据或复杂训练技术的情况下，这种方法在 7B 基线上的绝对提升达到了 10%。

#### Embedding head designs

![[Pasted image 20260127165159.png]]

文章使用最后n层作为embedding head的初始化 ，然后再做finetune

#### Hard Negative Mining

![[Pasted image 20260123151118.png]]

在通过对比学习训练鲁棒嵌入模型时，另一个主要挑战是选择硬负样本，即语义上与目标相似但并非真实匹配的样本。作者采用基于聚类的硬负采样方法来解决这个问题。首先，使用批内负样本训练嵌入模型，然后使用该模型为训练集生成嵌入。对于每个查询，根据嵌入相似性对候选样本进行排序，并选择排名前 k 的非匹配候选样本形成排序矩阵。最后，应用聚类算法构建包含硬负样本的批次，用于重新训练和增强嵌入模型。我们观察到采用硬负采样方法后，各项任务性能提升了高达 4%。

我发现论文的内容和博客有点出入，两个我都截取了部分内容放上来
## QQMM-embed

[Improve Multi-Modal Embedding Learning via Explicit Hard Negative Gradient Amplifying](https://arxiv.org/pdf/2506.02020)

![[Pasted image 20260123155422.png]]

![[Pasted image 20260123155434.png]]

### Method
#### Gradient Analysis of info-NCE
为了增大hard negative的效果，最直接的方法就是放大hard negative的梯度

![[Pasted image 20260123160332.png]]

![[Pasted image 20260123160443.png]]


#### Explicit Gradient Amplifier

![[Pasted image 20260123160732.png]]

作者认为决定负样本是否难以识别的关键因素不是它与查询的绝对相似度，而是它与查询的相似度与正样本的相似度相比有多接近。于是定义了一个指标，对比正样本计算负样本的难度Relative Similarity based Hardness score (RS-H)

![[Pasted image 20260123161334.png|300]]


根据前面的公式，我们可以看出损失函数的权重是p，然后为了放大hard negative的效果，可以修改p的值，从而影响gradient

![[Pasted image 20260123162209.png|300]]

![[Pasted image 20260123162253.png]]

这就是模块的整个步骤，但是其中我有些不理解的意思是$\bar{P}-I$，是意思是这个权重矩阵最后整体要加上一个$I$矩阵的意思吗，应该是加上来正样本的值

## [Seed1.6-Embedding](https://seed1-6-embedding.github.io/)

![[Pasted image 20260123162805.png]]

seed1.6-flash架构

### Training Method
1. Stage1: Text Continue Training
2. Stage 2: Multimodal Continue Training
3.  Stage 3: Fine-Tuning

## RzenEmbed-v1-7B

没有做什么太多创新

方法上是进行了一些负样本的优化

![[Pasted image 20260128155036.png]]


训练的几个阶段
1. Multimodal Continual Training
2. Fine-Tuning
3. Task-Specific Learnable Temperature
    1. τ越小，分布越锐利，迫使模型关注最难处理的负样本；而τ越大，分布越平滑，促使模型更均匀地考虑所有负样本。


## e5-omni

多模态带来的三个问题：
- 依赖于模态的锐度：使用单一全局温度可能会导致某些模态组合的 logits 过于尖锐，而其他模态的 logits 过于平缓，这会造成对比梯度的不平衡；
- 负样本难度不平衡：负样本难度在不同模态间存在差异，且在训练过程中会发生变化；当将所有批内负样本一视同仁时，优化可能会被许多非常简单的负样本主导，从而削弱后期的学习信号，并限制混合模态批次中的细粒度区分能力；
- 排名不稳定：异构输入之间的不一致几何结构使得排名对分数的小幅变化非常敏感，即使所有项目都嵌入在同一个空间中也如此。

文章提出的三种方法：
- Modality-aware Temperature Calibration. 引入了一个轻量级校准模块，该模块使用可训练的pre-模态缩放向量来校准跨模态的相似性对数值。这种自适应重缩放有助于平衡不同模态组合下的对比训练信号。
- Controllable Negative Curriculum. 使用基于分位数的阈值选择批内负样本，并在预热期后逐步增加难度。为了减轻负样本选择引入的偏差，我们进一步加入了去偏对比目标（Chuang 等，2020），这可以稳定混合模态批次中的优化过程。
- Batch Whitening and Covariance Alignment. 对嵌入应用批量白化变换（Ermolov 等，2021），并添加 CORAL 风格的正则项（Sun 和 Saenko，2016）以对齐跨模态的小批量协方差。这在共享空间中协调了二阶几何特征，并使对各种全模态输入的相似性比较更加一致。

![[Pasted image 20260128160550.png]]

简单来说从图里理解可以知道
Modality-aware Temperature Calibration：温度是基于输入模型做了一个重新的计算

![[Pasted image 20260128162908.png|300]]

Controllable Negative Curriculum：把比较easy的negative，mask掉，还有一个DCL，debiased constant

![[Pasted image 20260128162826.png|300]]

Batch Whitening and Covariance Alignment：使得query和target的一阶和二阶统计特征靠近，即均值和cov，为此引入loss

![[Pasted image 20260128184156.png|300]]
![[Pasted image 20260128184218.png|300]]

最后在可视化部分对这里做了一个验证

![[Pasted image 20260128184248.png]]

## GVE

Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum

因为这个是视频为主的，我只看里面部分的内容

![[Pasted image 20260129110511.png]]

### Modality Pyramid

Modality Pyramid 不是一个模型结构，而是一套“训练课程（curriculum）+ 动态任务调度”的策略，用来在高度异构的多模态检索任务中，避免简单任务过早主导训练、复杂任务学不会，从而学到更可泛化的统一 embedding。

因为目前训练数据都非常的杂，什么形式的任务都有，容易简单的任务占据了大多数的优化，这里的主要方法就是说通过设置一个模型探针，在每个epoch开始之前判断每个任务的学习程度，然后据此采样，加了一个sigma参数，使得一开始是简单的样本多，后来越来越多加入有难度的样本

![[Pasted image 20260129111820.png]]

## UME-R1

这篇文章特别一些地方是用了RL在里面，用的GRPO

![[Pasted image 20260129151234.png]]

数据做了一些清洗与采样

  

Reward分为几个部分：

1. Format Reward。如果按照prompt里面说的格式遵守则得到1，否则0

2. Embedding Reward. 分成两个部分，一个是正样本和负样本的排序reward，还有一个是正负样本similarity gap的reward

![[Pasted image 20260129151404.png]]

## ReMatch

整体来说在模型的使用上进行一些改进，首先在生成embedding的时候，在最后多加了几个learnable token，用多个embedding来表示一个文件（用loss限制尽量正交），然后在判断是否相关的时候将原始文本和embedding都一起送入MLLM做两两对比（多个embedding做一个平均，再过一个MLP）

![[Pasted image 20260129160132.png]]

