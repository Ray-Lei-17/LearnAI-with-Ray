最近强化学习在大语言模型的训练当中起到了非常重要的作用，也显示出了很强大的效果，但是很多时候文章介绍只是放一个公式，具体的方法就不介绍了，很难懂，最近向GPT老师请教后，有了初步的概念，尝试进行一个整理，对于我自己也是一个梳理，也希望提供给有需要快速了解的人。

## 强化学习的基础概念

强化学习是让一个“智能体”（agent）在某个“环境”（environment）里学会做决策的过程。

> 简单说：**Agent 与环境交互，得到反馈（奖励），并学习如何获得更多奖励。**

比如：
- 下围棋的 AI（环境是棋盘，动作是落子）
- 自动驾驶的算法（环境是道路，动作是转向/加速）
- LLM 的 RLHF 阶段（环境是对话上下文，动作是生成下一个 token）

### 基本要素

| 名称                           | 含义            | 举例（对话场景）               |
| ---------------------------- | ------------- | ---------------------- |
| 状态（State, $s_t$)             | 当前情境          | 用户输入的 prompt           |
| 动作（Action, $a_t$)            | Agent 的行为     | 模型输出的一个 token          |
| 策略（Policy, $\pi(a,s)$)       | Agent 选择动作的规则 | 在状态下采取各动作的概率分布         |
| 奖励（Reward, $r_t$)            | 环境给出的反馈信号     | 用户是否喜欢这段回答             |
| 价值函数（Value function, $V(s)$) | 状态的“未来收益”期望   | 从当前上下文继续写下去，最终能获得多好的奖励 |
| 折扣因子（Discount, $\gamma$)     | 衡量未来奖励的重要性    | 通常 < 1，用来让长期奖励不会无限叠加   |


### 强化学习的目标：最大化总奖励

![[Pasted image 20251013112709.png]]
在这个公式中的折扣因子不会影响最优策略，但是使用折扣因子会让训练变得更加稳定。

### Reward v.s. Value function

在上面的基础概念当中，我们会发现两个很相似的概念，“奖励”和“价值函数”，理清楚这两个概念有助于我们后面的理解，GPT老师是如下解释的：
![[Pasted image 20251013115400.png]]
![[Pasted image 20251013115431.png]]
![[Pasted image 20251013115503.png]]
![[Pasted image 20251013115522.png]]

### 动作价值函数$Q^\pi(s_t,a_t)$

从GPT老师的讲解中，我们除了厘清奖励告诉你“现在做得怎么样”和价值告诉你“未来还能多好”之外，我们还会关注到有两个核心函数：“状态价值函数”和“动作价值函数”，关于状态价值函数我们已经进行了一定的了解，然而这个动作价值函数，我们还尚未了解，我对动作价值函数进行了进一步的了解
![[Pasted image 20251013120819.png]]
![[Pasted image 20251013120851.png]]
![[Pasted image 20251013120914.png]]
![[Pasted image 20251013120932.png]]
![[Pasted image 20251013120947.png]]

# 强化学习方法分类

至此我们已经基本完成了所有基础要素概念的引入，接下来我们将对强化学习的方法进行了解。正如上一章GPT老师所说，由这些基础衍生出来了两种类型的强化学习方法：Value-based方法和Policy-based方法。现在目前LLM使用的方法基本是Policy-based的，但是进行基础的介绍有益于总体的理解，所以我们下面对于各个类型的方法都进行一下基本的介绍。

## Value-based

![[Pasted image 20251013161306.png]]
正如前序所说，value-based方法不直接对于策略进行学习，而是对于这个动作价值函数进行学习。而智能体进行动作的时候，直接只用greedy的方式进行动作的选择。
![[Pasted image 20251013171203.png]]
对于函数的更新，我们给出两种方法的例子，一个是TD，一个是Monte Carlo，它们两个一个很大的区别是TD是每走一步都能进行更新，而Monte Carlo是在一个episode结束获得一整个轨迹之后再进行更新。

#### TD, Temporal Difference 时序差分

时序差分的关键概念在于，我们每走一步之前我们会利用价值状态函数或者动作状态函数对于未来的奖励有一个估计，而每次我们实际走了这步之后我们会获得实际上的奖励，于是就产生了一个时序差分的误差，我们就通过这个差值对我们的函数进行更新。
![[Pasted image 20251013175339.png]]
![[Pasted image 20251013175403.png]]
看懂了这个流程，大概就理解了时序差分的核心。下面进行一些简单的扩展，没有需要的话可以跳过就行了，不影响后面内容的理解。
![[Pasted image 20251013175955.png]]
![[Pasted image 20251013180101.png]]

> **例子**
> ![[Pasted image 20251013214633.png]]![[Pasted image 20251013214657.png]]![[Pasted image 20251013214716.png]]![[Pasted image 20251013214733.png]]![[Pasted image 20251013214748.png]]

#### Monte Carlo 蒙特卡洛

而Monte Carlo正如我们刚刚所说，是得到完整的轨迹之后再进行更新的。
![[Pasted image 20251013180356.png]]
![[Pasted image 20251013180417.png]]
![[Pasted image 20251013180431.png]]

> **例子**
> ![[Pasted image 20251013214943.png]]![[Pasted image 20251013215008.png]]![[Pasted image 20251013215037.png]]![[Pasted image 20251013215055.png]]

## Policy-based

介绍完了value-based，我们回到policy-based的内容上，与value-based不一样的是，policy-based的关注点在于策略的直接学习，最后直接得到一个用于执行的策略。

### REINFORCE

我们再回到强化学习的目标：
![[Pasted image 20251013211051.png]]
REINFORCE针对这个累计奖励的概率直接计算梯度进行更新
![[Pasted image 20251013211144.png]]
![[Pasted image 20251013210050.png]]
![[Pasted image 20251013210132.png]]
稍微强调一下回报$G_t$和策略目标函数 $J(\theta)$之间的区别
![[Pasted image 20251013212234.png]]
![[Pasted image 20251013212315.png]]

> **例子**
> ![[Pasted image 20251013232142.png]]![[Pasted image 20251013232218.png]]![[Pasted image 20251013232239.png]]![[Pasted image 20251013232257.png]]

## Mixed

由于前述所说的REINFORCE的框架的缺点，所以将前面policy-based中的概念以价值网络critic的形式引入训练，来提供更平滑的信号。

### Actor–Critic

![[Pasted image 20251013225052.png]]
![[Pasted image 20251013225135.png]]
在这个地方进行参数更新的时候，除了TD误差，在Actor网络更新的时候引入不同的优势函数进行该步动作与平均奖励的差异

#### Advantage 优势函数

优势函数通过“哪个动作比平均好”来决定梯度方向。如果只用 $Q(s,a)$，那不同状态下的奖励尺度差别很大，梯度会非常 noisy。引入 V(s) 之后，你相当于在每个状态上都减去了一个 baseline，使得更新方向更稳定、更低方差。
![[Pasted image 20251013231447.png]]
![[Pasted image 20251013231507.png]]
![[Pasted image 20251013231530.png]]
![[Pasted image 20251013231552.png]]
![[Pasted image 20251013231609.png]]

### PPO

针对`PPO`而言，其关键点在于更新`Actor`的时候使用的函数进行了改进。
![[Pasted image 20251013232026.png]]
在PPO的公式中除了优势函数，又引入了一个新老策略的比值的概念，后面的clip就是对这个比值范围的限制，防止一次更新的步子过大。

#### 重要性采样$r_t​(\theta)=\frac{\pi_{old​}(a_t,​s_t​)}{π_\pi​(a_t​,s_t​)}​$
##### 原始解释

![[Pasted image 20251013233329.png]]
![[Pasted image 20251013233348.png]]
![[Pasted image 20251013233420.png]]
##### 解释更新
上面的解释当时给了我一种错觉，`PPO`是一种`off-policy`的策略，但实际上`PPO`是一个`on-policy`的方法，或者说近似`on-policy`，虽然会迭代采样比较新的方法，但是中间还是会有一些轮次是在最近采样的轨迹上更新的，并不代表完全最新的策略，所以还是存在策略重要性比值

下面讲讲`off-policy`和`on-policy`的方法的区别
###### `on-policy` v.s. `off-policy`
![[Pasted image 20251107104933.png]]
![[Pasted image 20251107104953.png]]
![[Pasted image 20251107105020.png]]

上面能看到`on-policy`和`off-policy`的主要区别就在于用的数据是新还是旧，然后同时我们还可以有一个发现，就是`off-policy`的方法多是`value-based`的，而`on-policy`的多是`policy-based`的，GPT老师是这么给我解释的

![[Pasted image 20251107110438.png]]

| 方法类型             | 核心目标                                  | 为什么倾向某种策略                                                                      |
| ---------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
| **Value-based**  | 学习 $Q(s, a) = \mathbb{E}[R\|a,s]$的估计值 | 因为 Q-learning 通过自举（bootstrapping）更新，目标动作来自当前策略，但数据可以来自任何策略，所以天然支持 `off-policy` |
| **Policy-based** | 直接学习策略 $\pi_\theta(a\|s)$             | 因为策略梯度的期望是基于当前策略分布计算的，必须用当前策略采样的数据，所以通常是 `on-policy`。                          |

![[Pasted image 20251107110509.png]]

| 算法                   | 类型           | On/Off Policy       | 说明                                |
| -------------------- | ------------ | ------------------- | --------------------------------- |
| **Q-learning / DQN** | Value-based  | ✅ Off-policy        | 因为可以用经验回放                         |
| **SARSA**            | Value-based  | ✅ On-policy         | 因为更新目标用的是当前策略的动作                  |
| **REINFORCE**        | Policy-based | ✅ On-policy         | 需要当前策略轨迹来估计梯度                     |
| **A2C / A3C**        | Actor-Critic | ✅ On-policy         | 策略梯度部分需要当前数据                      |
| **DDPG / TD3 / SAC** | Actor-Critic | ✅ Off-policy        | 用 replay buffer，融合 value-based 思想 |
| **PPO / TRPO**       | Policy-based | ✅ On-policy（约束更新步长） | 限制策略变化太大时的 instability            |

![[Pasted image 20251107110521.png]]

###### 核心公式
![[Pasted image 20251107110656.png]]
![[Pasted image 20251107110709.png]]

#### 轮次的概念

由于PPO是近似`on-policy`，让我对轮次的概念产生了疑惑，GPT老师的表述中每一轮PPO会重新采样一次数据
![[Pasted image 20251107111911.png]]
![[Pasted image 20251107111925.png]]
![[Pasted image 20251107111952.png]]

#### 整理

整体的思想我们已经了解，我们再把上面的内容按照执行的步骤进行一些重新的整理：
![[Pasted image 20251014115111.png]]
![[Pasted image 20251014115142.png]]
![[Pasted image 20251014115216.png]]
![[Pasted image 20251014115302.png]]

##### REINFORCE → Actor-Critic → Importance sampling → PPO
![[Pasted image 20251107111052.png]]
![[Pasted image 20251107111115.png]]
![[Pasted image 20251107111147.png]]
#### REINFORCE v.s. PPO
先看看REINFORCE的公式

![[Pasted image 20260114170717.png|400]]

然后我们看看PPO的公式

![[Pasted image 20260114171219.png|600]]

可以看到在REINFORCE里面有一个log，而PPO中没有，于是我找gpt老师问了一下，很有收获，总的来说，log的出现是为了将期望
![[Pasted image 20260114175106.png]]
![[Pasted image 20260114175554.png]]
![[Pasted image 20260114175713.png]]
![[Pasted image 20260114175943.png]]
![[Pasted image 20260114180021.png]]
![[Pasted image 20260114180102.png]]
![[Pasted image 20260114180121.png]]
![[Pasted image 20260114180218.png]]
![[Pasted image 20260114180238.png]]
![[Pasted image 20260114180303.png]]
![[Pasted image 20260114180340.png]]
![[Pasted image 20260114180503.png]]
![[Pasted image 20260114180531.png]]
![[Pasted image 20260114180559.png]]
![[Pasted image 20260114180625.png]]
# RL for LLM

现在对于RL中的PPO算法有了一定的了解，那它又是如何用来训练大语言模型的呢？

## PPO

![[Pasted image 20251014161206.png]]
![[Pasted image 20251014161313.png]]
![[Pasted image 20251014161333.png]]
![[Pasted image 20251014161413.png]]
![[Pasted image 20251014161502.png]]
在这里我产生了一个一整个句子该如何更新的疑问
![[Pasted image 20251014161522.png]]
![[Pasted image 20251014161635.png]]
![[Pasted image 20251014161650.png]]
![[Pasted image 20251014161712.png]]
![[Pasted image 20251014161800.png]]
![[Pasted image 20251014161828.png]]
GPT老师给出了一个伪代码如下
```python
# ------------------------------------------------------------
# 初始化
# ------------------------------------------------------------
policy = LLM()                 # 当前策略 πθ
old_policy = copy(policy)      # 旧策略 πθ_old，用于计算 ratio
value_net = ValueNetwork()     # 价值函数 Vϕ
reward_model = RewardModel()   # 奖励模型 Rψ
optimizer_policy = Adam(policy.parameters())
optimizer_value = Adam(value_net.parameters())

# 超参数
gamma = 0.99                   # 折扣因子
lam = 0.95                     # GAE 参数
clip_eps = 0.2                 # PPO clip范围
K_epochs = 4                   # 每次采样后更新次数

# ------------------------------------------------------------
# 训练循环
# ------------------------------------------------------------
for iteration in range(num_iterations):
    
    # === Step 1: 采样 ===
    batch_prompts = sample_prompts(batch_size)

    trajectories = []
    for prompt in batch_prompts:
        # 用旧策略生成回答（不更新梯度）
        with torch.no_grad():
            response, logprobs, values = old_policy.generate_with_values(prompt)
            reward = reward_model.score(prompt, response)
        
        trajectories.append({
            "prompt": prompt,
            "response": response,
            "logprobs": logprobs,   # 每个token的 log π_old(a|s)
            "values": values,       # 每个token的 Vϕ(s)
            "reward": reward        # 句级奖励
        })

    # === Step 2: 计算 Advantage ===
    # 奖励是句级（整段），但要传播到每个token
    all_advantages, all_returns = [], []
    for traj in trajectories:
        R = traj["reward"]
        values = traj["values"]
        # 使用 GAE 估计每个时间步的优势
        advantages, returns = compute_GAE(R, values, gamma, lam)
        all_advantages.append(advantages)
        all_returns.append(returns)
    
    # === Step 3: PPO 更新 ===
    for _ in range(K_epochs):
        for traj, advantages, returns in zip(trajectories, all_advantages, all_returns):
            # 当前策略的 logprob
            logprobs_new, values_new = policy.evaluate(traj["prompt"], traj["response"])
            
            # ratio: 新旧策略概率比
            ratio = torch.exp(logprobs_new - traj["logprobs"])
            
            # PPO-Clip 损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Value loss
            value_loss = torch.mean((returns - values_new) ** 2)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            loss.backward()
            optimizer_policy.step()
            optimizer_value.step()
    
    # === Step 4: 更新旧策略 ===
    old_policy.load_state_dict(policy.state_dict())

```

![[Pasted image 20251014162059.png]]
![[Pasted image 20251014162114.png]]
![[Pasted image 20251014162132.png]]

## DPO

参考资料：[# 详细推导DPO算法](https://zhuanlan.zhihu.com/p/697757566)
本站简单拆解：[[DPO]]

简述：不再额外保留reward网络和价值网络，直接使用两两样本的对比来优化模型


总体思路：利用Bradley-Terry建模关系，利用优化目标， 解出最优reward的闭式解，带入回Bradley-Terry公式中，得到loss的形式
![[Pasted image 20251014163043.png]]
![[Pasted image 20251014163103.png]]
![[Pasted image 20251014163148.png]]
![[Pasted image 20251014163242.png]]
![[Pasted image 20251014163256.png]]


## GRPO

简述：**组内成员之间的两两比较**得出的**相对胜率**作为奖励信号
![[Pasted image 20251014162818.png]]
如果前面的内容都能看懂的话，结合后面部分，这个图应当是很容易看懂了，我们先把两个公式晒出来
![[Pasted image 20260114161233.png]]
![[Pasted image 20260114161315.png]]
![[Pasted image 20260114161354.png]]
![[Pasted image 20260114161437.png]]
![[Pasted image 20260114161739.png]]
![[Pasted image 20260114161756.png]]
![[Pasted image 20260114161813.png]]
然后我们直接转问DeepSeek：
![[Pasted image 20251014162258.png]]
![[Pasted image 20251014162349.png]]
![[Pasted image 20251014162411.png]]
![[Pasted image 20251014162433.png]]
![[Pasted image 20251014162450.png]]

### 与DPO的联系与区别

![[Pasted image 20251014162604.png]]
![[Pasted image 20251014162638.png]]
![[Pasted image 20251014162650.png]]
![[Pasted image 20251014162702.png]]
![[Pasted image 20251014162716.png]]

### leave-one-out strategy
`leave-one-out strategy`是指在计算优势函数的时候，把自己这个样本摘出去，这样是为了减少计算时的方差，GPT老师解释如下

![[Pasted image 20251107112340.png]]
![[Pasted image 20251107112431.png]]
![[Pasted image 20251107112441.png]]

## DAPO
简述：把全对或者全错的样本删掉，因为梯度为0，根本不需要做计算更新

![[Pasted image 20251107112617.png]]
![[Pasted image 20251107112638.png]]
##### 公式
![[Pasted image 20251107112719.png]]![[Pasted image 20251107112936.png]]
###### 几个关键部分
![[Pasted image 20251107113907.png]]![[Pasted image 20251107113931.png]]
##### 相比GRPO的改进点
![[Pasted image 20251107114112.png]]
![[Pasted image 20251107114125.png]]

### DUPO
tongyi的web sailor里面提供的方法，简述而言是吧DAPO里面去掉的样本用同batch中其他的样本填充来加快速度
###### 原文
![[Pasted image 20251107114539.png]]
###### GPT回答
![[Pasted image 20251107114637.png]]
![[Pasted image 20251107114654.png]]
![[Pasted image 20251107114711.png]]
![[Pasted image 20251107114747.png]]
这个就是DAPO去除梯度为0的样本的公式
![[Pasted image 20251107114802.png]]
![[Pasted image 20251107114852.png]]
## GSPO
简述：把重要性采样给换成基于组的
![[Pasted image 20251104213526.png]]
### 回答
![[Pasted image 20251107115333.png]]
![[Pasted image 20251107115348.png]]
### 序列级别
![[Pasted image 20251107115943.png]]
![[Pasted image 20251107115959.png]]
![[Pasted image 20251107120024.png]]
![[Pasted image 20251107120036.png]]

# 总结对比
![[Pasted image 20251107114942.png]]