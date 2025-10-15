## DPO公式的推导
这个公式我已经推了两次，但是由于没有地方记录，导致下次看又忘记了，准备整理放在这里，方便我下次使用。其实有一个知乎文章写的很好了，我主要也就不再重写了。

[# 详细推导DPO算法](https://zhuanlan.zhihu.com/p/697757566)

总体的思路：将reward从Loss中推导成一个closed-form的表达式，然后代入到对比两个样本的BT模型中，把复杂的reward部分消掉

首先我们来看如何比较两个样本的好坏，这里用到Bradley-Terry模型
![[Pasted image 20251015102614.png]]
然后我们回到传统的RLHF的目标函数
![[Pasted image 20251015103020.png]]
这个函数将KL散度展开，可以得到策略$\pi_\theta$的闭式解
![[Pasted image 20251015103209.png]]
![[Pasted image 20251015103425.png]]
由此得到
![[Pasted image 20251015103223.png]]
接下来我们将这个$r_\phi(x,y)$式子代入到Bradley-Terry的公式中就可以得到DPO的损失函数，将包含$r_\phi(x,y)$的$Z(x)$消掉
![[Pasted image 20251015103605.png]]
由此即可得到DPO的Loss