AI的发展实在是太快了，每天总有新的概念输出，学习应接不暇，但当我进行AI的深度写作之后，我发现这个协作方式真的非常重要，AI使用中有很多的坑，容易学的越来越蠢，目前解说的文章很多，但是很多我感觉质量良莠不齐，所以我在这里自己整理一下，抓出其中的重点

## [OpenAI原博客](https://openai.com/zh-Hans-CN/index/harness-engineering/)

1. 不能一次性直接给一个超级大的AGENTS.md文件记所有的东西，上下文会很快腐烂
2. 代码仓库的知识库位于一个结构化了的 `docs/` 目录中
3. 简短的 `AGENTS.md`（大约 100 行）被注入到情境中，主要用作地图
4. **渐进式披露**：智能体从一个小而稳定的切入点开始，并被指导下一步该去哪里查看，而不是一开始就被淹没
5. 为 Codex 提供更多情境意味着要组织和展示正确的信息，比如让智能体重新实现部分功能子集比绕过公共库中不透明的上游行为更便宜。就是不import，直接重新造轮子

下面是参考结构

```
AGENTS.md
ARCHITECTURE.md
docs/
├── design-docs/
│   ├── index.md
│   ├── core-beliefs.md
│   └── ...
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   ├── new-user-onboarding.md
│   └── ...
├── references/
│   ├── design-system-reference-llms.txt
│   ├── nixpacks-llms.txt
│   ├── uv-llms.txt
│   └── ...
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
├── PRODUCT_SENSE.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```

##  [OpenAI 说的 Harness Engineering，到底是什么？](https://mp.weixin.qq.com/s/NTeOqAT9AbuGtrW09_xDHw)

按照openai的原文重新解读了一遍，经过ta整理后稍微梳理更清楚了几点
1. 让AI有反馈机制
2. 这并不是一种可以轻易复制的状态。它依赖非常具体的仓库结构、工具链和长期投入。意思这不是一个容易的工作，要长期构建，长期进化
3. 技术债像垃圾清理。编写规则后，自动的定时对于仓库内的垃圾代码进行清理

> 这也提醒我们，未来好的工程管理能力，很可能会越来越像：
> 
> •能不能把品味写成规则
> 
> •能不能把规范写成检查
> 
> •能不能把技术债处理做成持续运行的系统


> 如果用一句话概括 OpenAI 这篇文章，我会写成：
> 
> AI 写代码只是开始，真正决定效率上限的，是你有没有为智能体设计好工作环境。
> 
> 这，就是 harness engineering。

## [【Harness Engineering】同一件事，第三次被命名](https://mp.weixin.qq.com/s/SQGRI_DbFDOLjP904QuOPQ)

整体思想：**agent 犯了错，你就工程化一个方案让它不再犯。**

感觉内容有误，错误描述OpenAI方案为写在一整个Agents.md里面，不再仔细点评这个文章

## [技术教科书：顶级开发团队设计的Harness工程项目源码什么样](https://mp.weixin.qq.com/s/MKWckXraK1irNvMgCIJXZw)

> 近期，某顶级 AI Agent 研究团队的一个工业级 Harness 项目源码在开发者社区中引起广泛关注。这个项目是一个基于 TypeScript 的 CLI 形态 AI Coding Agent，其工程规模和架构成熟度令社区印象深刻：

> _"REPL.tsx 单文件 875KB，我以为我看错了小数点。这不是代码，这是一部长篇小说。"_ — HN 评论

文章开篇这么说，反应了半天，笑出来了，说的是Claude Code

工程构建的一些优化我就不看了，主要看一些比较有关的部分

### Part 4: 查询引擎 — Agent Loop 的核心
> 
> ![[bbfdf1fd24cbca3433674773232bd7dd.png]]
> 
> ![[b5a2968e252ec7bfe9b18b6ddbca005f.png]]
> 
> **Level 1 — Snip Compact**：基于标记的历史裁剪。在消息流中找到 snip 边界标记，移除标记之前的消息。最轻量，无需 API 调用。
> 
> **Level 2 — Micro Compact**：缓存编辑压缩。利用 API 的 cache editing 能力，在不破坏整体缓存的情况下删除特定工具调用的结果。
> 
> **Level 3 — Context Collapse**：上下文折叠。将多轮工具调用结果折叠为摘要，但保留结构。这是一个**读时投影**——折叠视图在每次发送前重新计算，原始消息仍然保存在 REPL 的完整历史中。
> 
> **Level 4 — Auto Compact**：全量摘要压缩。当上下文接近窗口限制时，使用 LLM 生成对话摘要替换历史消息。这是最重的操作，但也是最后的防线。

这个地方如果我没记错，有人发现了bug，自动压缩的重度模式只于时间有关，会导致即使完全没动，token也会被大量的消耗


### Part 5: 多 Agent 编排与任务系统
> 
> ```
> # Python 伪代码重构 -- 展示核心设计思路  
> type TaskType =  
>     | 'local_bash'            # Shell 命令（后台进程）  
>     | 'local_agent'           # 本地子 Agent（独立进程）  
>     | 'remote_agent'          # 远程 Agent（WebSocket 连接）  
>     | 'in_process_teammate'   # 进程内队友（共享内存）  
>     | 'local_workflow'        # 本地工作流脚本  
>     | 'monitor_mcp'           # MCP 监控任务  
>     | 'dream'                 # "梦境"任务（后台分析）  
>   
> type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'killed'
> ```
> 
> 当 `AGENT_COORDINATOR_MODE=1` 时，主线程变成**协调器**，只负责分配任务，所有实际工作由 worker Agent 完成
> 
> `DreamTask` 是一个独特的任务类型——它在后台运行分析任务，类似于模型在"做梦"。这可能用于：代码理解、依赖分析、或预测性的上下文准备。
> 
### Part 7: Harness Engineering — 从该项目看 2026 年最热工程范式
> 
> 而当我们深入该项目的 512K 行源码时，惊讶地发现——这不是一个"应用了 Harness Engineering 理念的项目"，而是**这个理念最完整的工业级实现**。它的代码中，模型调用相关的部分不到 5%，剩下 95% 全部是 Harness。
> 
> **范式演进的三部曲**：

|阶段|时间|核心问题|类比|
|---|---|---|---|
|**Prompt Engineering**|2023-2024|如何让模型理解你的意图|对马喊话的技巧|
|**Context Engineering**|2025|如何给模型正确的知识边界|给马看地图|
|**Harness Engineering**|2026-|如何让 Agent 可靠、持续、不失控|造高速公路，配护栏和限速牌|
> 
> Harness Engineering 的核心哲学用八个字概括："人类掌舵，Agent 执行"**。它不试图让模型"变聪明"，而是通过工程手段，让一个"已经很聪明但不可预测"的模型在**约束和反馈中稳定工作。
> 
> _"Agent 的每一次失败，都是环境设计不完善的信号。正确的回应不是换更强的模型，而是重新设计它运行的环境。"_ — Cassie Kozyrkov

下面关于Harness Engineer的六大支柱我觉得挺好的，我就不赘述了，我直接把原文贴上来

####  7.2 六大支柱在 该项目中的完整落地

> 综合 OpenAI、某头部 AI 实验室、Martin Fowler、LangChain、Latent Space 和 Cassie Kozyrkov 六方文献，Harness Engineering 可以提炼为**六大工程支柱**。下面我们逐一解析每个支柱在 该项目源码中的具体实现——这不是抽象理论，而是从 512K 行生产代码中提取的工程实践。
> 
>
##### 支柱一：上下文架构 🗺️

> 
> **核心理念**：精准设计进入模型上下文的信息。研究表明，当上下文窗口利用率超过 40% 时，模型推理质量显著下滑。
> 
> **该项目的实现是教科书级的**——它构建了一条完整的**四级压缩管道**（详见 Part 4.3），从轻量裁剪到全量摘要，渐进降级：

```
Snip Compact → Micro Compact → Context Collapse → Auto Compact     零API调用     缓存编辑         读时投影          LLM摘要     (最轻量)                                       (最重，最后手段)
```

> 但这只是压缩端。注入端同样精心设计：
> 
> - **分层记忆系统**：`PROJECT.md` 自动加载 → `memdir/` 持久化知识库 → Session 记忆（自动失效）→ Repo 级知识
>     
> - **按需注入**：技能系统通过 `SkillTool` 按需发现和注入，而非启动时全量加载
>     
> - `getEffectiveContextWindowSize()`：动态计算可用上下文窗口，为摘要预留 `min(maxOutput, 20000)` tokens（基于 p99.99 数据 17,387 tokens）
>     
> 
> 这与 LangChain Deep Agents 的策略高度一致——LangChain 在工具结果超过 20,000 tokens 时卸载到文件系统，该项目则通过 Context Collapse 在读时投影为摘要视图。两者殊途同归：**永远不要让上下文窗口变成垃圾场**。
> 
>
##### 支柱二：架构约束 ⛓️

> 
> **核心理念**：用代码和工具强制执行规则，而非依赖 prompt 的"软约束"。依赖模型"自律性"是不可靠的。
> 
> 该项目在这个支柱上的投入极重——整个权限系统就是一个五层纵深防御体系：
> 
> ![五层权限安全模型](https://mmbiz.qpic.cn/sz_mmbiz_png/KVER9adz904qMrE1MtAR0yRux1biaUvBMgGRFAjXoycIAib2z2LeTFLmdO4OXyKdUNFMdrDQj4fiahSic4eE8zDbbXMUyicibBu7piap1Fm1gpNEHM/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=9)
> 五层权限安全模型
> 
> 层层递进：**Deny Rules（不可见）→ Tool-level Permissions（自检）→ Generic Rules（规则匹配）→ Permission Mode（模式判断）→ Auto Classifier（分类器兜底）**。
> 
> `buildTool()` 工厂函数的**Fail-Closed 默认值**是这个支柱最精髓的体现：

```
# Python 伪代码重构 -- 展示核心设计思路DEFAULTS = {    "is_concurrency_safe": lambda _: False,  # 假设不安全    "is_read_only": lambda _: False,          # 假设会写入    "is_destructive": lambda _: False,}
```
> 
> 忘了设置？那就走最受限路径。**遗漏不是漏洞**。
> 
> OpenAI 的方法是用确定性 Linter 强制执行层级依赖，该项目的方法是用 Pydantic Schema + 编译时 `is_feature_enabled()` 特性开关 + 五层权限模型。两者的共同点是：**用机器约束代替人的自律。**
> 
>
##### 支柱三：自验证循环 🔄

> 
> **核心理念**：在执行流程中内置验证检查点，防止死循环与静默失败。
> 
> 这是 该项目源码中最被低估的设计。`query()` 的 `while(true)` 循环有 **16 个步骤**，其中只有步骤 8 是"调用模型"，其余 15 个全是验证和修复逻辑：

```
# Python 伪代码重构 -- 展示核心设计思路# 简化的 query() 循环结构whileTrue:    # 1-2: 前置预取与预算（技能/工具结果）    # 3-6: 上下文预处理（Snip/Micro/Collapse/Auto 压缩）    # 7: 阻塞限制检查    # ★ 8: 调用 API（唯一的模型交互！）    # 9: 流式工具执行    # 10: 后采样 Hooks（stop_hooks 验证）    # 11: 中断处理    # 12: 停止 Hooks（含 max_tokens 恢复）    # 13: Token 预算检查    # 14: 附件消息注入（记忆/技能/命令队列）    # 15: MCP 工具热更新    # 16: transition 追踪（记录"为什么继续"）
```

> `transition` 字段是验证循环的精华——它不是一个调试工具，而是一个**可断言的状态机**：

```
# Python 伪代码重构 -- 展示核心设计思路CONTINUE_REASONS = [    "next_turn",                  # 正常下一轮    "max_tokens_recovery",        # 截断恢复（含 attempt 计数）    "reactive_compact_retry",     # 反应式压缩重试    "collapse_drain_retry",       # 折叠排空重试    "stop_hook_blocking",         # 停止 Hook 阻塞    "token_budget_continuation",  # Token 预算续传]
```

> 测试可以直接断言"这次循环是因为 max_tokens 恢复才继续的"，而不需要从消息内容中反向推导。这与 Birgitta Böckeler 提出的"生成者与评估者分离"高度契合——`stopHooks` 系统允许注入外部验证逻辑，让用户或外部系统充当"评估者"角色。
> 
##### 支柱四：上下文隔离 🧊

> 
> **核心理念**：多 Agent 协作时保持每个 Agent 的上下文纯净，防止跨边界信息污染导致级联故障。
> 
> 该项目在这个支柱上有三层隔离设计：
> 
> 1. **进程级隔离**：`AgentTool` 生成的子 Agent 拥有完全独立的上下文窗口、消息历史和 `AbortController`。子 Agent 的错误不会传播到父级（query.ts 不会因为子 Agent 出错而终止回合）
>     
> 2. **通信接口化**：Agent Swarms 中的 Teammates 通过 `SendMessageTool` 传递**结构化消息**，而非共享原始上下文。Unix Domain Socket (UDS) 保证了 ~50μs 的通信延迟（vs HTTP 的 ~500μs）
>     
> 3. **Coordinator 模式的控制面/数据面分离**：
>     

```
# Python 伪代码重构 -- 展示核心设计思路# 协调器线程只有 3 个工具coordinator_tools = [AgentTool, TaskStopTool, SendMessageTool]# Worker 有完整的工具集worker_tools = [BashTool, FileReadTool, FileEditTool, FileWriteTool, GrepTool, GlobTool, ...]
```

> 协调器**不能自己动手**——它只负责分配任务和检查结果。这与微服务架构中"控制面不处理数据"的理念完全一致。该团队在《Effective Harnesses》中也推崇类似的双层 Agent 架构（初始化 Agent + 编码 Agent），该项目则把它做到了 6 种 TaskType 的完整体系。
> 
>
##### 支柱五：熵治理 ♻️

> 
> **核心理念**：对抗系统状态的自然熵增——随着任务执行，上下文变得越来越混乱，记忆碎片化，文档腐烂。
> 
> 这是该项目最前卫的设计所在——**AutoDream 梦境系统**（详见 Part 8.2）本质上就是一个**自动化熵治理引擎**：

|熵治理手段|该项目实现|触发条件|
|---|---|---|
|**上下文蒸馏**|`/compact`<br><br> + Auto Compact|手动或上下文接近窗口限制|
|**知识沉淀**|`memdir/`<br><br> 持久化写入|Agent 主动调用|
|**状态清理**|Session 记忆自动失效|会话结束|
|**后台整合**|AutoDream 4 阶段|24h + 5 sessions 双重门控|
|**碎片整理**|Dream Phase 4: Prune & Index|AutoDream 最后阶段|

> OpenAI 用"后台清洁 Agent"自动偿还技术债务，该项目的 AutoDream 做的是同样的事——但对象不是代码，而是 AI 的记忆。它借鉴了认知科学中的**记忆巩固理论**：人类在 REM 睡眠阶段重播白天经历，将短期记忆转化为长期记忆。AutoDream 对 AI 做了同样的事。

##### 支柱六：可拆卸性 🔌

> **核心理念**：模块化设计，使 Harness 能随模型迭代优雅适配。防止与特定模型深度耦合。
> 
> 该项目的可拆卸性体现在三个层面：
> 
> 1. **依赖注入**：`QueryDeps` 只有 4 个字段（`callModel`, `microcompact`, `autocompact`, `uuid`），用函数签名类型引用保持类型同步。替换模型只需替换 `callModel` 一个字段
>     
> 2. **Skills = Markdown**：技能定义不绑定任何特定模型或 API。一个 Markdown 文件可以在不同模型上通用，因为它定义的是**流程**而非**调用方式**
>     
> 3. **MCP 标准协议**：外部工具通过 Model Context Protocol 接入，独立于 该项目的内部实现。MCP 工具可以用任何语言编写，运行在独立进程中
>     
> 4. **模型降级容错**：当主模型过载时，自动切换到 fallback 模型，strip thinking 签名块防止 400 错误
    

![五层能力扩展体系](https://mmbiz.qpic.cn/sz_mmbiz_png/KVER9adz905fb8r0VTDM5p2WHbynlr8WkQ31CpczBk7aCCCkuZdWzmWNFFoyoWYe1PHrGiasWo4WKMB83XiaUPzjnYuwEMpxSORqkBNM1xMOM/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1#imgIndex=10)
五层能力扩展体系

### Takeaway

我抽取几点有益的点

- **把 95% 的精力放在 Harness 上**：模型调用只是冰山一角。压缩、权限、隔离、恢复、熵治理——这些才是决定 Agent 可靠性的关键
- **熵治理要自动化**：别指望"定期手动清理"——AutoDream 式的后台整合才是可持续方案
- **Skills = Markdown 是最佳扩展模式**：零代码门槛，不绑定模型，可版本化管理

## [Harness Engineering是什么？为什么Harness来了，也得用混合检索？](https://mp.weixin.qq.com/s/fVkwW_xJ5wFqoD2VzFGfWA)

主要讲OpenAI博客里的内容，还有一些补充内容来自Rajasekaran我记录一下

> Agent对自己的评估偏差，比很多人想的要大。最常见的两个失效模式与应对，分别如下：
> 
> 失效模式1：Context Anxiety（上下文焦虑）
> 
> 随着 Context Window 逐渐填满，Agent 会出现提前收尾任务的情况。不是因为任务完成了，而是因为它感觉自己的上下文窗口快到上限了，担心无法继续接收信息。
> 
> 行业内最常见的应对方式是 Compaction（上下文压缩）：摘要历史对话和信息，让同一个 Agent 在压缩后的 Context 里继续工作。这种方式能保住任务的连续性，但无法消除 Agent 的焦虑感，毕竟压缩的是历史信息，没有压缩快撑不住了的心理预期。
> 
> 另一种解决方案是 Context Reset（上下文重置）：彻底清空当前 Agent 的上下文，启动一个全新的 Agent 实例，用结构化的 handoff artifact（交接文件）传递前任的状态和待办事项。这种方式能消除焦虑，但代价是交接文件必须足够完整，否则新的 Agent 无法无缝衔接，会导致任务中断。
> 
> 这两种策略本质上是在连续性和清醒度之间选边。Claude Opus 4.5 基本消除了 Context Anxiety 这个行为本身，使得这个版本的 Harness 可以直接去掉 Context Reset，交给 SDK 的自动 Compaction 处理。

> 失效模式2：自我评估偏差
> 
> Agent 评估自己产出时，倾向于给虚高分。
> 
> 这种情况在主观任务上最明显——UI 设计没有等价的单元测试可以客观验证。但即使是有明确正误标准的代码任务，这个问题同样存在：它会先发现问题，然后说服自己"这其实没那么严重"，最终通过了本不该通过的检查。
> 
> Rajasekaran 从 GAN 里借了一个思路：把做事的 Agent 和评判的 Agent 彻底拆开。
> 
> 调教一个独立的 Evaluator 让它保持怀疑，远比让Generator 批判自己的作品容易得多。Evaluator 被单独调教为天生挑剔的审查者，一旦外部评判存在，Generator 就有了具体的迭代目标，而不是在自我满足里原地打转。

后面说的是文档多了之后如何去搜索文档，所以提到向量检索

他最后的一些观察有点意思

> 看到这里，我们已经梳理了 Harness Engineering 的核心定义、两个经典落地案例，以及关键的技术支撑。但还有一个更重要的规律，藏在这些案例背后：Harness 里的每个组件，都编码了一个模型自己做不到的假设——而这些假设，会随着模型能力的提升逐渐失效。
> 
> 比如，Sprint 分解结构在 Claude Opus 4.5 版本中是必要的，因为当时的模型在长任务中会失去连贯性，需要通过 Sprint 拆分来保证任务推进；但到了 Opus 4.6 版本，模型能力大幅提升，这种分解结构就变成了多余的负担，可以直接删除。
> 
> 再比如，Context Reset 是针对早期模型Context Anxiety的补偿机制，当模型本身消除了这种焦虑，这个组件就可以从 Harness 中移除。
> 
> 这意味着，Harness Engineering 不是一个固定不变的框架，而是需要随模型能力持续重新校准的动态系统。每次新模型发布，Harness Engineer 要做的第一件事，就是重新审视整套 Harness 系统，找出那些不再是承重墙的组件，把它们拆掉——因为这些组件的存在，反而会增加系统的冗余，降低 Agent 的运行效率。
> 
> 这个逻辑同样适用于 Context Retrieval 这一层。随着模型长Context 能力增强，检索-注入机制的粒度和时机都会变化。昨天必须精细管理的 Context Slice，明天可能可以整页直接塞进去。
> 
> 说到底，基础设施配置是和模型能力一起演化的变量。Harness 里任何一个必要的组件，都在等待被更聪明的模型证明为多余。

## [Qoder 工程实践：Harness Engineering 指南](https://mp.weixin.qq.com/s/Et3WwNtEXEgxjaQHrQFDyQ)

这篇文章更偏向于工程实践，我觉得还是蛮有参考的价值

> Prompt 写得再好，也没法穷尽代码库的所有隐式规则。Harness 工程的思路不一样：与其教 Agent 怎么做，不如让它自己验证做得对不对。靠代码、linter、测试来保证正确性，而不是靠 LLM 的"直觉"。这些机械化检查不会出错，不会遗忘，也不会被上下文压缩掉。就像 CI/CD 对人类开发者的作用——自动拦截问题。只不过这次拦截的时机更早，不是合并前，而是写代码前。

### 关键原则

> 1. 首先，仓库是唯一的事实来源。Aone里的讨论、钉钉会议上的口头约定、架构师脑子里的蓝图，这些对 Agent 来说都不存在。不在仓库里，Agent 就看不见；看不见就会违反。所以第一步是把一切编码到仓库中：架构决策、层级约束、命名规范。不是写在 Wiki 里，不是发在群里，而是作为版本化的文件提交到 Git。这样知识跟着代码一起走，新人 clone 仓库就能拿到全部上下文，Agent 打开项目就能读到一切。
> 2. 但编码到仓库不意味着把所有东西塞进一个文件。很多团队的第一反应是写一份巨大的 AGENTS.md，500 行，什么都有。问题是：当一切都重要时，什么都不重要。500 行的指令文件挤占了 Agent 宝贵的上下文窗口，留给实际任务的空间反而少了。AGENTS.md 应该是地图，不是手册——控制在 ~100 行，只做索引和指路，详细内容放在 `docs/` 目录里按需加载。要改 auth 模块？先读 AGENTS.md 找到路，再读 `docs/design-docs/auth.md` 拿细节。用不上的文档根本不加载。保持短小精悍还有一个好处：不容易腐烂，巨大的指令文件会迅速过时。
> 3. 然后是约束的粒度。Harness 不规定"你必须用这个设计模式"或"函数必须这样写"，它只管架构边界。大多数代码库的包和模块之间存在自然的依赖方向：类型定义被所有人 import、业务逻辑依赖类型但不依赖 HTTP 层、HTTP handler 依赖业务逻辑。Harness 把这种自然方向编码为层级编号——Layer 0 是类型定义（不 import 任何内部包），Layer 1-2 是工具函数和配置（只依赖更低层），Layer 3 是业务逻辑，Layer 4+ 是 HTTP handler 和 CLI 命令。规则就一条：高层可以 import 低层，反过来不行。在这个边界之内怎么实现，随便。跟管理大型平台团队一样：中心化约束，本地自治。
> 4. 最后一条原则关乎人的角色。以前是人写代码、AI 辅助补全；现在反过来了——人设计系统（架构、约束、验证规则），Agent 在系统内执行。人的价值从"写出正确的代码"变成了"设计出让 Agent 能可靠产出正确代码的环境"。你不再需要自己拧每一颗螺丝，但你得确保流水线是对的。


### 具体落地：两个角色--协调者和执行者

> harness-creator 负责分析代码库、生成基础设施（文档、lint 脚本、目录结构）；harness-executor 在这套基础设施中执行开发任务。它们的关系很简单——executor 启动时先看 AGENTS.md 在不在，不在就自动喊 creator 来搭，搭完再继续干活。所以任何项目都能直接用 executor 起步。

> creator 首次运行时会审计项目现状，按文档覆盖率、lint 规则覆盖率等维度打出 0-100 分。0-20 分基本是裸奔状态，从零搭建全套；21-70 分说明有基础但有缺口，针对性补充；71 分以上已经比较健康，微调就行。
> 
> executor 的工作流是：检测环境 → 加载上下文 → 制定计划 → 人类批准 → 执行 → 验证 → 完成。注意"人类批准"这一步不是走过场——对于非简单任务，executor 会创建一个执行计划文件，包含任务目标、影响范围、分阶段步骤、验证方式和回退策略。你扫一眼就知道 Agent 打算怎么干，觉得不对可以直接改方向。
> 
> 一个装备了 Harness 的项目，大致长这样：

```
my-project/
├── AGENTS.md                    ← 导航地图（~100行）
├── docs/
│   ├── ARCHITECTURE.md          ← 架构、层级、依赖规则
│   ├── DEVELOPMENT.md           ← 构建/测试/lint 命令
│   ├── PRODUCT_SENSE.md         ← 业务上下文
│   ├── design-docs/             ← 组件设计文档
│   └── exec-plans/              ← 执行计划（active / completed）
├── scripts/
│   ├── lint-deps.*              ← 层级依赖检查
│   ├── lint-quality.*           ← 代码质量规则
│   ├── verify/                  ← 端到端功能验证
│   └── validate.py              ← 统一验证管道
├── harness/
│   ├── tasks/                   ← 任务状态和检查点
│   ├── trace/                   ← 执行轨迹和失败记录
│   └── memory/                  ← 经验教训存储
└── [业务代码...]
```


> 每一部分在执行过程中都有用：文档提供知识，脚本提供验证，harness/ 提供状态管理和学习能力。其中 scripts/ 下面的机械执法层是核心——它把团队约定从"希望被遵守"变成"不遵守就报错"。


### 先验证

显式规则描述+事先验证

> Harness 里的验证分几类。最核心的是依赖方向检查（lint-deps）——`core/` 不能 import `ui/`，`api/` 和 `cli/` 不能互相引用。对应的就是前面说的层级规则，靠扫描源码里的 import 语句来检查。其次是质量规则检查（lint-quality），强制一些编码规范：单文件不超过 500 行，禁止 `console.log` / `print()`（要求用结构化日志），禁止硬编码品牌字符串。这些规则说起来都是常识，但 Agent 不提醒就不会遵守。lint 脚本有 Go、TypeScript、Python 三个版本，creator 会根据你的项目类型自动生成。
> 
> 但重点不是事后检查，而是事前预防。Agent 通常的工作流是写代码、跑测试、发现错误、修复、再跑测试。当一个层级违反在 50 行代码写完后才被 linter 抓到，修复代价很大——撤销改动、重新设计，差不多要消耗 10 次 tool call。而如果在写代码前先问一句"这样做合法吗"，两次交互就够了：

高级操作要先验证，报错信息要详细，提供统一验证入口

> 不是每个操作都需要预验证。改个函数体不需要，加个测试文件不需要。但只要涉及"在新位置创建文件"或"添加跨包 import"，就该先验证。层级违反是 Agent 翻车的头号原因。
> 
> 这里有一个容易被忽视但影响很大的细节：linter 的错误信息质量直接决定了 Agent 能不能自愈。一条 `Forbidden import in core/types/user.go` 看完不知道怎么办；但如果改成 `core/types/user.go imports core/config (Layer 0 → Layer 2). Layer 0 packages must have NO internal dependencies. Fix: Move config-dependent logic to a higher layer, or pass the config value as a parameter.`——什么规则违反了、为什么是问题、怎么修，全在里面。一条好的报错本身就是一次教学。
> 
> Harness 不鼓励 Agent 直接跑 `go build` 或 `go test`，而是提供统一验证入口 `validate.py`，因为"验证通过"在每个项目里含义不同。验证顺序有讲究：先 build（编译都不过就别往下了），再 lint-arch（架构约束），然后 test（功能正确性），最后还有一步容易被漏掉——verify。

功能层面的验证要verify，进行实际操作的尝试

> build + lint + test 能覆盖大部分问题，但有一类问题它们抓不到：功能层面的正确性。测试通过了不代表功能是对的——测试本身可能覆盖不全，或者 Agent 写的测试恰好绕过了关键路径。verify 步骤是项目级别的端到端功能验证——不是"函数返回值对不对"，而是"用户执行这个操作，最终结果对不对"。比如一个 CLI 工具，verify 会实际运行几个典型命令检查输出；一个 Web API，会发几个真实请求验证响应。
> 
> 很多项目一开始并不具备端到端 verify 的能力。这时候 Harness 会引导你创建 verify skill——一组针对项目的可复用验证脚本。思路是先识别核心用户路径（比如"创建用户→登录→查看资料"），然后把每条路径编码成可执行的验证脚本。creator 分析项目时会自动生成骨架放在 `scripts/verify/` 下面，你填充具体的断言逻辑就行。这让验证闭环从"代码能跑"提升到了"功能正确"。


![[Pasted image 20260410153432.png]]

错太多就别让AI干了，故意添加错误代码

> 如果同一个错误转了 3 圈还没过，就别让 Agent 继续挣扎了，停下来交给人。
> 
> 几条踩过坑之后的经验：故意引入违规来测试 lint，加一个跨层 import 确认能被检测到，如果 lint 没报错说明护栏是纸糊的；永远不要禁用 lint 规则来"解决"问题，Agent 有时候会想绕过规则（比如注释掉 lint 配置），应该改代码而不是改规则；测试不需要每次跑全量，executor 支持只跑受影响的包，速度快很多。

### 珍惜上下文

> 中等复杂度以上的任务，协调者绝不写代码。

![[Pasted image 20260410153610.png]]

协调者不写代码，执行者从干净上下文开始，快速改一下也不行

> 解法是把 Agent 拆成两层。协调者（Coordinator）只管规划、委派、汇总，一行代码都不碰；执行者（子代理）每次从干净的上下文开始，拿到精确的 prompt，干完就释放。子代理不知道之前发生了什么，但它拿到的 prompt 里包含了它需要知道的一切。任务完成后，详细上下文被丢掉，协调者只保留一段摘要。信息经过了压缩和筛选，而不是无差别地堆在上下文里。
> 
> 破坏这个原则最常见的方式是"只是快速改一下"。协调者发现一个小问题，心想"这么简单的东西不用启子代理，我直接改了"。一次编辑变成了五次（因为牵扯出其他地方），五次变成二十次，上下文就被消耗殆尽了。这个陷阱看起来很合理——"就改一行而已嘛"——但你低估了代码的牵连性。如果你发现协调者正在用 Edit 或 Write 工具修改源代码，立刻停下来，启动子代理。没有例外。

复杂度不同的任务用不同能力的模型，做到最佳分配

> 复杂度不同，执行方式也不同。改个 typo、加行日志这种小事可以直接做；多文件一致性修改就该委派给子代理；到了重构、新模块这种结构性变更，子代理还得在 Git Worktree 里隔离执行——相当于仓库的临时副本，成功了合并，失败了丢掉，不污染主分支。快速判断方法：能用一句话描述且不包含"和"字的，直接做；需要清单来跟踪改了哪些地方的，委派；需要做设计决策和权衡的，委派加隔离。
> 
> 委派子代理时还有一个容易被忽略的杠杆：不是所有任务都需要用最强的模型。一个"重命名变量"的任务和一个"重构认证模块"的任务，对模型能力的要求完全不同。前者要的是快和便宜，后者要的是深度推理。如果所有子代理都用同一个顶级模型，既浪费钱又浪费时间。
> 
> 协调者在委派时可以根据任务性质指定模型。快速执行类（改 typo、简单重命名）用轻量模型如 Claude Haiku，响应快、成本低；深度推理类（复杂重构、架构级变更）用 GPT-5.3 Codex 或 Claude Opus，质量远比速度重要；代码检索类（在大型代码库中定位相关文件）用 Gemini 3 Flash，速度第一。

要用不同模型交叉review

> 一个中等复杂度的功能开发，可能会同时调度三四个不同模型的子代理：一个用 Flash 检索相关代码，一个用 Opus 做核心实现，完成后再用 Codex 做交叉 review。协调者本身用中等模型就行——它不写代码，只做调度。总成本可以降低 60-70%，复杂任务的质量不打折扣。
> 
> 说到交叉 review，这是整个流程中一个很值得展开的环节。机械化验证（lint、test、verify）能抓到规则层面的问题，但抓不到逻辑合理性——代码编译通过、测试全绿、架构合规，但实现思路可能有隐患：竞态条件、边界遗漏、不必要的复杂度、命名让人看不懂。这些东西只有"另一双眼睛"才能发现。
> 
> 关键词是"不同模型"。如果用同一个模型既写代码又 review，它容易对自己的产出"视而不见"——类似于人写完文章自己检查总觉得没问题。换一个架构和训练数据都不同的模型来 review，思维盲区的重叠就小得多。
> 
> 实际操作中，交叉 review 嵌入在子代理完成编码、机械验证通过之后，协调者接受结果之前。Review 子代理拿到变更的文件 diff、任务描述和架构文档，检查逻辑正确性、边界条件、命名清晰度、潜在性能问题，输出"通过"或"需修改"（附具体建议）。需要修改就回到编码子代理修复，再过一轮 review。

错误的问题记录到trace中，犯错太多就写成新规则

> 不是所有任务都需要交叉 review。简单修改直接过机械验证就够了。但涉及核心业务逻辑、安全相关代码、或者影响面较广的重构，加一轮是值得的。成本上，review 子代理只需要读 diff 和文档，一次 review 的成本大约是编码成本的 10-20%。
> 
> 这样做还有一个副作用：review 中发现的问题会被记录到 `harness/trace/` 里。如果某类问题反复出现，说明应该把它编码成新的 lint 规则——把 review 中的"软知识"逐渐转化成验证管道中的"硬规则"。这也是 Harness 自我进化的一部分，后面会讲到。
> 
> 中等以上的任务还会设检查点。每完成一个阶段、跑过验证就存档，包括已有的架构决策。任务中断了，下一个 Agent 能从检查点恢复，不会做出跟前面矛盾的选择。检查点携带架构决策这一点很重要——没有它，新 Agent 可能走一条完全不同的路，引入微妙的不一致。

### 进化

![[Pasted image 20260410160230.png]]

Critic定时分析失败案例，失败次数过多的记录成的规则，而成功率高的路径也可以记录为轨迹

> 每次验证失败都被结构化地保存到 `harness/trace/failures/`。Critic 脚本定期分析这些记录，找出模式和根因——比如发现 `internal/cache` 被 7 次违规 import，根因是它没被加入层级映射表，建议修复是把它加入 Layer 1。然后 Refiner 根据建议去更新 Harness：把遗漏的包加入 linter 规则，改写含糊的错误信息，补上缺失的文档。整个循环跑起来就是：Agent 执行 → 验证抓到问题 → Critic 分析模式 → Refiner 更新规则 → 下一个 Agent 受益。
> 
> Harness 还维护着三种"记忆"。情景记忆记录具体事件和教训，比如"macOS 下 /var 是 /private/var 的符号链接，会导致工作区路径比较失败"——这类教训可能只要 10 秒加载，却能省下一整个重试循环。程序记忆记录成功的操作步骤，比如"添加 API 端点的标准五步流程，成功率 90%"，新的子代理执行同类任务时会先查这里。失败记忆专门供 Critic 分析用。每次任务开始时，executor 会查询相关记忆，这些记忆的价值随着项目演进不断累积。
> 
> 这套闭环走到极致会出现一个很有意思的东西：轨迹编译。当同一类任务被成功执行了三次以上，而且步骤高度一致——比如"添加 API 端点"每次都是创建类型文件、写服务方法、加 handler、注册路由、写测试——这个模式就可以被"编译"成一个确定性脚本，像 `make add-endpoint NAME=foo` 这样。以后再做同类任务直接跑脚本，LLM 都省了。脚本失败了再回退到 Agent 模式就好。
> 
> 这是个棘轮效应：每一个被编译的成功模式都变成了永久基础设施，Agent 被释放去处理真正需要创造力的新问题。整个系统的运行成本越来越低，能力越来越强。

### 开始实践

> 理论讲完了。好在 Harness 不是全有或全无的——哪怕不搭完整的六层基础设施，一个 AGENTS.md 就能让 AI 协作体验好一截。这套方法适用于任何能跑 shell 命令的 coding agent（Qodercli、Claude Code、Codex 等都行）。多人协作、有明确分层的中大型项目收益最明显；个人项目或原型阶段，一个 AGENTS.md 加个简单的 lint 脚本就够了。
> 
> 新项目最简单——告诉 creator 你要做什么，它会问几个基本问题，然后直接生成全套基础设施。老项目也不难，creator 会扫描代码库、分析 import 关系、推断层级映射，生成反映代码现状的文档。随着项目演进，定期跑一次 Improve 模式做体检：creator 输出审计报告，指出哪些包没被 linter 覆盖、哪些文档没跟上代码变化，然后 executor 来实施修复。

```
# [项目名] Agent Guide
## 快速链接
- [架构总览](docs/ARCHITECTURE.md) — 分层规则、数据流
- [开发指南](docs/DEVELOPMENT.md) — 构建、测试、lint 命令
## 构建命令
make build      # 构建项目
make test       # 运行测试
make lint-arch  # 运行架构 lint
## 分层规则
Layer 0: types/         → 纯类型定义，无内部依赖
Layer 1: utils/         → 工具函数，仅依赖 Layer 0
Layer 2: config/        → 配置，依赖 Layer 0-1
Layer 3: core/services/ → 业务逻辑，依赖 Layer 0-2
Layer 4: api/ cli/ ui/  → 接口层，依赖 Layer 0-3，彼此不互相引用
## 质量标准
- 结构化日志，禁止 console.log / print()
- 单文件不超过 500 行
- PascalCase（类型）、camelCase（函数）、kebab-case（文件名）
```

> 为什么叫 AGENTS.md？这个文件名业界的标准（https://agents.md/），Agent 打开项目时会自动查找并读取它，作为了解项目的起点。类似于 README.md 是给人看的，AGENTS.md 是给 Agent 看的。建议的节奏：今天先把 AGENTS.md 建出来；这周加一个 lint-deps 脚本把层级规则定下来；这个月搭完整的验证管道；之后开启 Critic → Refiner 反馈循环，让 Harness 跟着代码一起长。最佳实践是先用 creator 把 Harness 建到 70 分以上，再开始用 executor 日常开发。
> 
> 回到开头那个场景。装了 Harness 之后，同样的任务会变成这样：Agent 启动，读 AGENTS.md 找到相关文档；列出执行计划，你扫一眼批准；子代理开始写代码，每个结构性操作前先跑预验证；层级违反在写代码前就被拦住了；完成后另一个模型的子代理做交叉 review，抓出机械验证发现不了的逻辑问题；每个阶段存检查点、跑验证；任务做完，经验教训记下来，下一个 Agent 接着用。
> 
> Agent 不需要更聪明——它只是能看见更多了。
> 
> 说到底，环境设计的投入回报远高于 prompt 调优。一套好的 Harness 能让普通模型产出可靠的代码，而没有 Harness 的顶级模型照样会在同样的坑里反复栽。搭建的前期成本不高——一个下午就能建好基本的 AGENTS.md 和 lint 脚本。但它的价值会随着时间复利式增长：记忆越来越丰富，lint 规则越来越完善，越来越多的操作模式被编译成确定性脚本。半年后回头看，你的仓库已经变成了一个高度适配你们团队工作方式的 Agent 运行环境，任何新加入的人（或新会话的 Agent）都能立刻进入状态。
> 
> 竞争优势不再是 Prompt，而是 Trajectory。这些积累，换个模型复制不来。