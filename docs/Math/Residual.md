这篇是介绍关于Residual Connection相关的工作的
## HYPER-CONNECTIONS

简述来说，hyper connection是depth connection再加width connection的组合。根据expansion rate n每一层保存n个hidden state，DC是把output和这些hidden state加权相加，WC将多个hidden state加权相加作为input输入layer
![[Pasted image 20260122204042.png]]
### Method

![[Pasted image 20260121171536.png]]
![[Pasted image 20260121171754.png]]
![[Pasted image 20260121172201.png]]
![[Pasted image 20260121173033.png]]
DC和WC有一部分共同作用的参数$diag(A_r)$

上面这种是静态的超链接，还有动态的，动态的概念就是将前面的矩阵里面的参数变成一个简单网络预测出来的值
![[Pasted image 20260121173451.png]]
![[Pasted image 20260121173701.png]]

### WHY HYPER-CONNECTIONS
这个部分分析Hyper connection的作用
![[Pasted image 20260121174028.png]]
![[Pasted image 20260121174043.png]]
Pre-Norm和Post-Norm都能够写成Hyper connection的特殊形式
![[Pasted image 20260121183601.png]]
![[Pasted image 20260121183620.png]]
![[Pasted image 20260121183640.png]]
因为存在多个hidden state同时向后传输，那么我们通过HC的参数矩阵的设计可以选择让模型实际上实现类似parallel或者sequential的效果，实现丰富的层间信息交互方式。

### Results

![[Pasted image 20260121185449.png]]
![[Pasted image 20260121185526.png]]
随着n增大，效果明显提升
![[Pasted image 20260121184719.png]]
讨论了一下SHC、DHC在几种情况下的比较
![[Pasted image 20260121190046.png]]
![[Pasted image 20260121190100.png]]
![[Pasted image 20260121190127.png]]
![[Pasted image 20260121190153.png]]
![[Pasted image 20260121190358.png]]
#### Visualization
![[Pasted image 20260121223842.png]]
下面是得到的一些结论
![[Pasted image 20260122112122.png]]
总体来说我认为这篇工作相当有意思，对residual连接的修改，通过增加每一层的hidden state数量，引入了block更多结合作用的可能性，但从实验数值来看，效果没有想象中那么多，但也还行吧，有些可惜。

## mHC: Manifold-Constrained Hyper-Connections
作者提到前面的HC这种多样化方法虽然能大幅提高性能，但却从根本上损害了残差连接固有的identity mapping特性，从而导致严重的训练不稳定性和可扩展性受限，而且还会产生显著的内存访问开销。为此ta们提出mHC框架，将 HC 的残余连接空间投影到特定流形上以**恢复identity mapping属性**，同时结合严格的基础设施优化以确保效率。
![[Pasted image 20260122155711.png]]
回顾一下Residual原始的公式
![[Pasted image 20260122161346.png|300]]
下面是Hyper Connection的公式
![[Pasted image 20260122170928.png |500]]
将HC推导到多层得到
![[Pasted image 20260122201309.png|550]]
这些没有限制的mapping H可能导致unbound放大或减弱，影响训练的稳定性。作者提出mHC框架将 HC 的Residual连接空间投影到特定流形上以**恢复identity mapping属性**。具体来说，mHC利用Sinkhorn-Knopp算法（Sinkhorn and Knopp, 1967）将$\mathcal{H}^{res}_l$熵投影到Birkhoff多面体上。

下面是原文的简略介绍，里面提到了不少细节，还是可以注意看看。
![[Pasted image 20260122202323.png]]

### Preliminary
![[Pasted image 20260122203421.png]]
![[Pasted image 20260122203651.png]]
这里有一个值得注意的小实验
![[Pasted image 20260122203826.png]]
这三个矩阵，对应前面的文章应当是：
- $\mathcal{H}^{post}_l$对应$\mathcal{B}$
- $\mathcal{H}^{res}_l$对应$\mathcal{A}_r$
- $\mathcal{H}^{pre}_l$对应$\mathcal{A}_m$
然后经过消融发现，这三个矩阵里面$\mathcal{H}^{res}_l$是效果最好的，也就是说hidden state之间的信息交流更加重要

下面是另一个实验
![[Pasted image 20260122204915.png]]
前面说的犹豫破坏了residual的identity mapping性质，无界的mapping会导致训练的稳定性被破坏
于是ta们提出了一个Amax Gain Magnitude来检测，可以看到Fig2的(a)中在12k的时候loss出现了一个上升，但是由于是和mHC为基准的，不好说是mHC下降的太快了，还是训练出现了不稳定。

Amax指标的计算方式是这样的：有两个指标。第一个指标基于复合映射$\prod^{L-l}_{i=1} \mathcal{H}^{res}_{L-i}$行和的最大绝对值，捕捉前向传递中最坏情况下的扩展。第二个指标基于列和的最大绝对值，与后向传递相对应。

然后作者研究了一下系统开销，分析表明，HC增加内存访问成本的因子大约与n成正比。
![[Pasted image 20260122210424.png]]

### Method
#### Manifold-Constrained Hyper-Connections
简单来说，是把$\mathcal{H}^{res}_l$限制成一个双随机矩阵，即行列都被归一化为概率分布的矩阵（由此行/列和分别为1）
![[Pasted image 20260122211432.png]]![[Pasted image 20260122211451.png]]
这个双随机矩阵的空间也叫做Brikhoff Ploytope，这个空间有一些优异的性质
- 规范化，不超过1
- 闭包，乘起来也还在这个范围内
- 可以对应上几何解释，就相当于是permutation matrices的convex组合

#### Parameterization and Manifold Projection
original HC

![[Pasted image 20260123104116.png|300]]

mHC

![[Pasted image 20260123104446.png]]
不是特别理解post乘以2的原因是啥
#### Efficient Infrastructure Design (TODO)
### Experiments (TODO)