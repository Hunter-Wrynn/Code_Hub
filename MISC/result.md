[Research] MLLM Latent TTS
很多研究表明已经证明，当前多模态推理的token是disentangle，分为perception token和 reasoning token
https://arxiv.org/pdf/2603.25077
https://arxiv.org/pdf/2603.25077
因此在token-level更新的时候可以分开去更新
Introduction

近年来，多模态大语言模型（MLLMs）在视觉推理 benchmark 上取得了显著进展，但其错误类型仍然高度异质：有些错误来自模型没有正确感知图像，有些错误则来自模型已经看到了关键证据，却仍然选择了错误答案。这种不确定性在 test-time scaling 场景下进一步放大：额外的测试时计算虽然可能提升性能，但也可能放大语言先验，或者强化本来就错误的推理路径。

一个有前景的方向是 latent test-time scaling，即在推理阶段直接优化中间 hidden states，而不是重新训练模型。与单纯的输出空间搜索相比，latent refinement 能够直接作用于内部表示，因此提供了一种轻量、training-free 的样本级适配机制。然而，当前多模态 latent refinement 与 test-time scaling 所使用的监督信号仍然相当异质：有些方法依赖 answer-level 或 outcome-level signal，有些方法引入 internal confidence、process reward 或 perception-aware objective，但直接作用于 latent refinement 本身的显式 evidence-conditioned objective 仍然相对不足。在实践中，许多 reward-driven 变体仍主要受“答案在图像-问题条件下是否看起来 plausible”所支配，而不是直接检验该答案是否被可见证据支持。因此，这类 latent refinement 仍然容易被理解为一种更昂贵的 answer reranking。

这意味着问题的核心不一定在于 latent optimization 本身，而在于测试时优化所依赖的监督信号。对于多模态推理而言，一个有效的 reward 不应只判断最终答案是否 plausible，还应评估该答案是否 grounded in perceptual evidence，以及它是否相对于其他临近候选 获得了更强的证据支持。换言之，真正有效的 latent test-time scaling 应当显式耦合 perception 与 reasoning：perception 负责暴露图像中真正可见的证据，reasoning 负责偏向那些更能解释这些证据的答案假设。

基于这一观察，本文提出 Evidence-Calibrated Latent Test-Time Scaling（ECL-TTS），这是一个 training-free 的多模态 latent refinement 框架。ECL-TTS 首先提取一小组短的 visual clues，将其作为紧凑的感知证据。随后，它通过一个由三部分组成的 reward 来优化答案假设：(1) answer-level visual grounding，(2) relative clue consistency，即判断一个候选答案是否比其他候选更能解释这些 visual clues，以及 (3) answer margin，用于偏向相对更强的答案假设。进一步地，我们还研究了一个 Phase 3 扩展，引入 counterfactual crop verification，用于判断一个答案是否在局部支持视图上仍然成立，并在关键证据被移除时失去支持。

需要强调的是，我们并不将该方法表述为“改进了自由形式的 chain-of-thought 质量”。相反，我们将 reasoning 定义为 latent space 中基于证据的答案比较。在这一视角下，perception 的作用是提取紧凑的视觉证据，reasoning 的作用则是在这些证据约束下，对不同答案假设进行比较和 refinement。这样的 framing 更符合方法本身的结构，也更符合我们在实验中观察到的行为。

多个多模态推理 benchmark 上的实验结果支持了这一观点。作为主方法的 Phase 2 结合了 answer grounding、relative clue consistency 和 answer margin，在整体上给出了最稳定的收益。它将 Qwen2.5-VL 在 MathVista 上从 68.7 提升到 71.7，在 RealWorldQA 上从 63.66 提升到 67.71，在 HallusionBench 上从 70.56 提升到 71.71，并且在 MMStar、LogicVista 和 ScienceQA-IMG 上也带来了稳定增益。这些结果表明，相比只依赖 answer plausibility 或其他弱 grounding 的 latent-time objective，将 perception 与 reasoning 显式耦合到 reward 中是一条更有效的方向。
本文的贡献可以总结如下：
1. 我们指出了多模态 latent test-time scaling 的一个关键局限：尽管已有工作开始探索 process-aware 与 perception-aware supervision，但面向 latent refinement 的显式 evidence-conditioned reward 仍然不足，因此很难清晰地区分真正来自感知 grounding 的改进与 answer-level reranking。
2. 我们提出了 ECL-TTS，一个 training-free 的 latent refinement 框架，通过 answer grounding、relative clue consistency 和 answer margin 将 perception 与 reasoning 耦合起来。
3. 我们进一步提出了 counterfactual crop verification 扩展，用于在 latent refinement 中评估局部支持证据与 masked-evidence 敏感性。
4. 我们在多个多模态推理 benchmark 上验证了所提出的 evidence-calibrated reward 的有效性，并发现以 Phase 2 为代表的主方法在简单性与稳定收益之间提供了最好的权衡。

不更新模型参数，而是对每一个测试样本单独地编辑一段可控的生成 hidden states，并利用图像条件下的答案 likelihood，把模型输出推向更符合视觉证据的答案。
L主要由四部分组成：
1. 针对当前样本的 latent-state optimization
2. hidden state 与 token logit 两个空间上的双重正则
3. 围绕优化后 latent slice 的多候选探索
Method
设多模态模型在 (I, q) 上生成初始输出序列
$y = (y_1, y_2, \dots, y_T).$
在这一 rollout 过程中，我们记录生成 token 对应的最后一层 hidden states：
$h_1, h_2, \dots, h_T.$
沿用 latent refinement 的基本设定，我们选取一个可编辑的局部窗口
$z = [h_s, h_{s+1}, \dots, h_{s+L-1}],$
其中 s 是靠近答案区域的起始位置，L 是一个较短的更新长度。该 latent slice 用原始 hidden states 初始化，并在 test time 逐步优化。每一步更新后，我们从当前 latent 中解码多个候选输出，并利用 reward 对它们打分。
原始 latent refinement 的优化目标可以写为
$\mathcal{L}
= \mathcal{L}_{\text{policy}}
 + \mathcal{L}_{\text{logit}}
 + \mathcal{L}_{\text{anchor}},$

Visual Clue 提取
为了显式暴露感知证据，我们首先提取一个小规模的 visual clues 集合
$g = \{g_1, g_2, \dots, g_m\},$
其中每个 g_i 都是图像中直接可见的短原子事实，例如物体数量、相对位置、颜色或局部属性。这些 clues 不是完整的 chain-of-thought，也不是人工标注的 rationale，而是位于“图像”与“最终答案”之间的一层紧凑证据表示。
我们通过 prompting 让多模态模型在 latent optimization 之前先生成少量短的、图像可见的 observations，从而得到 g。这种做法有两个优点。第一，它不需要额外人工标注，就能引入一个显式的感知中间层。第二，它允许我们进一步判断：一个候选答案是否真的被可见事实支持，而不仅仅是“看起来像对的”。
Reasoning：证据约束下的答案比较
我们的关键设计，是不把 reasoning 建模为自由形式的 rationale 生成，而是建模为 perceptual evidence 约束下的 comparative answer-hypothesis refinement。设 a 表示从某个候选生成结果中抽取出的短答案，我们为每个候选赋予一个由 answer、clue 和 margin 共同组成的总分。
1 Answer Reward
我们保留 answer-level 的视觉 grounding 项：
$r_{\text{ans}}(a)
= \phi \big(
\log P(a \mid I, q)
- \lambda_q \log P(a \mid q)
\big),$
其中 \phi(\cdot) 是 squashing 函数，文本条件项用于压制 language prior。这一项保留了以往 latent refinement 中最稳定的信号：答案本身仍然必须在完整图像-问题条件下是合理的。
2 Relative Clue Consistency
一个强答案不仅自身分数要高，还应当更好地解释提取出的 clues。因此我们对 clue 集合在 candidate answer 条件下进行打分：
$s_{\text{clue}}(a, g)
= \frac{1}{m} \sum_{i=1}^{m}
\Big[
\log P(g_i \mid I, q, a)
- \lambda_g \log P(g_i \mid q, a)
\Big].$
与其直接将这个 clue 的绝对分数加到 reward 中，我们把它转换成 relative clue margin：
$\Delta_{\text{clue}}(a)
= s_{\text{clue}}(a, g)
- b_{\text{clue}}(a),$
其中 b_{\text{clue}}(a) 是由竞争候选构造出的 baseline clue score。最终的 clue 项定义为
$r_{\text{clue}}(a)
= \phi(\Delta_{\text{clue}}(a)).$
这种相对化处理非常关键。它避免了 clue signal 退化为所有候选共享的常数偏置，而真正变成区分不同答案假设的判别项。
3 Answer Margin
我们进一步引入相对答案优势项：
$\Delta_{\text{ans}}(a)
= s_{\text{ans}}(a)
- b_{\text{ans}}(a),$
其中 s_{\text{ans}}(a) 是未经 squashing 的 answer score，b_{\text{ans}}(a) 是最强竞争答案的分数。定义
$r_{\text{margin}}(a)
= \phi(\Delta_{\text{ans}}(a)).$
这一项鼓励 latent update 不仅提高某个答案的绝对得分，而且提高其相对于临近竞争答案的优势。
Result

Model	Method	MathVista	MMStar	LogicVista	RealWorldQA	HallusionBench	ScienceQA-IMG	MMMU val
Qwen2.5-VL-7B	CoT	67.80	62.00	44.52	63.66	70.56	89.74	50.76
	Self-consistency	70.90	63.47	42.51	65.49	68.77	89.60	50.76
	Best-of-N	71.60	63.93	42.73	68.37	70.56	90.63	52.04
	Reward-only	70.90	64.67	42.95	67.06	70.45	-	50.99
	LatentSeek	66.50	61.53	41.83	62.87	69.19	88.29	48.07
	Ours	72.10	64.73	46.98	67.45	71.71	90.73	51.23
InternVL3.5-8B	CoT	69.80	67.13	45.19	64.36	66.04	95.04	59.39
	Self-consistency	70.10	67.13	45.90	64.71	66.25	95.64	59.63
	Best-of-N	72.40	67.80	46.98	64.71	66.56	97.03	61.73
	Reward-only	71.30	68.07	46.53	65.10	66.61	96.98	59.63
	LatentSeek	-	-	-	-	-	-	-
	Ours	72.30	68.80	46.76	65.23	66.46	97.08	60.79


Qwen2.5-VL-3B	CoT	59.50	54.07	-	63.14	61.93	80.32	-
	Ours	62.20	56.13	-	64.05	62.88	81.51	-
InternVL3.5-4B	CoT	69.10	64.87	-	54.77	62.36	92.86	-
	Ours	70.70	65.13	-	63.53	64.04	95.19	-

Qwen2.5vl-32B
Benchmark	CoT	Ours	Link
MathVista	77.4	77.8	https://primus.alibaba-inc.com/19/training/type/stages/quark_rl/jobs/job-detail/primus2c1d44a88a67cfe9b718a9d98z
MMStar	67.9	68.0	https://primus.alibaba-inc.com/19/training/type/stages/quark_rl/jobs/job-detail/primusffbe847af83a6762f15882796z
RealWorldQA	67.8	69.7	https://primus.alibaba-inc.com/19/training/type/stages/quark_rl/jobs/job-detail/primus5d50c40c0b7a19bb38b0e126ez
HallusionBench	65.3	69.7	https://primus.alibaba-inc.com/19/training/type/stages/quark_rl/jobs/job-detail/primus78f3f484eb31707ac1e7ff799z
ScienceQA-IMG	93.2	93.6	https://primus.alibaba-inc.com/19/training/type/stages/quark_rl/jobs/job-detail/primus4730044b08fcce25e52fa85d0z
LogicVista	50.3	51.7	https://primus.alibaba-inc.com/19/training/type/stages/quark_rl/jobs/job-detail/primus51bd94f4f91752cf9ab666b40z
下一周计划

尝试更disentangle的perception与reaosning token tts，并且如何reweighting不同的reward
参考 https://arxiv.org/pdf/2603.25077

[Research] TR-OPD
Idea：
idea1:
在多模态推理强化学习中，不要把整条生成轨迹上的所有 token 都用同一种方式监督；而是把“perception”和“reasoning”拆成两个角色，在学生模型自己的 on-policy rollout 上，对不同角色的关键位置施加不同的选择性蒸馏信号。

更直接地说，这个方法希望解决一个长期存在的问题：
● 最终答案奖励太稀疏；
● 多模态任务里的错误来源混在一起；
● 统一的 RL 更新很难区分“没看对图”和“看对了但推理错了”；
● 统一的 OPD 又会把所有 token 一视同仁，无法体现角色差异。
因此，DRR-OPD 想做的是：
● 用角色分解perception和reasoning；
● 用 on-policy distillation 提供更稠密的过程监督；
● 用 selective masking 避免整段轨迹无差别蒸馏。

参考 https://arxiv.org/pdf/2603.25077 ， 先把一条rollout中的 perception token和 reaosning token 分类，然后分别用不同的teacher model对两种token做opd

进度： codebase已搭建好，测试token-level的opd稳定性

idea2:
在多模态推理中，将数学公式渲染成图片分别作推理往往效果差距很大，
可以用一种方式做studetn，另一种方式做teacher来opd
Result
[Agentic RL] Browser use 沙箱问题汇总
iAgent Toolcall 运行错误统计
指标	数量
总任务数	256
完整产出结果的任务数	256
正常完成任务数	200
完成但出现过 API/iAgent 问题的任务数	56
出现过 API 错误的任务数	1
出现过 iAgent 问题的任务数	55
正常完成 的定义：
● 任务有完整结果文件
● agent.log 中没有命中本报告定义的 API 错误模式
● agent.log 中没有命中本报告定义的 iAgent 错误/超时模式
API 错误
类型	任务数	出现次数
responses API 500 Internal Server Error	1	1
responses API 4xx	0	0
BadRequestError	0	0
iAgent 错误 / 超时
类型	任务数	出现次数
Navigation timeout, but page might still be usable	47	63
type 工具触发 NoneType 索引 TypeError	13	15
Task failed 标记	8	8
MCP 504 Gateway Time-out	5	5
工具 ExceptionGroup	3	9
工具 TimeoutError	3	3
MCP 404 Not Found	2	2
Sandbox 429	1	2
工具 McpError: Session terminated	1	1
按站点统计出现过问题的任务数
站点	有问题的任务数
Apple	5
ArXiv	8
Coursera	8
ESPN	35
API 错误明细
responses API 500 Internal Server Error
对应任务：
● taskCoursera--32
iAgent 错误明细
MCP 404 Not Found
对应任务：
● taskESPN--14
● taskESPN--17
MCP 504 Gateway Time-out
对应任务：
● taskApple--15
● taskESPN--19
● taskESPN--20
● taskESPN--21
● taskESPN--23
Sandbox 429
对应任务：
● taskESPN--25
工具 TimeoutError
对应任务：
● taskApple--15
● taskESPN--20
● taskESPN--23
工具 ExceptionGroup
对应任务：
● taskApple--15
● taskESPN--20
● taskESPN--23
工具 McpError: Session terminated
对应任务：
● taskESPN--17
工具 type 触发 TypeError: int() argument ... NoneType
对应任务：
● taskESPN--1
● taskESPN--10
● taskESPN--27
● taskESPN--28
● taskESPN--31
● taskESPN--33
● taskESPN--34
● taskESPN--35
● taskESPN--36
● taskESPN--39
● taskESPN--41
● taskESPN--42
● taskESPN--43
Navigation Timeout 提示
匹配的日志模式：
● Navigation timeout, but page might still be usable
对应任务：
● taskArXiv--10
● taskArXiv--11
● taskArXiv--12
● taskArXiv--17
● taskArXiv--19
● taskArXiv--22
● taskArXiv--30
● taskArXiv--37
● taskCoursera--0
● taskCoursera--16
● taskCoursera--21
● taskCoursera--22
● taskCoursera--26
● taskCoursera--28
● taskCoursera--38
● taskESPN--0
● taskESPN--1
● taskESPN--10
● taskESPN--13
● taskESPN--14
● taskESPN--15
● taskESPN--19
● taskESPN--2
● taskESPN--20
● taskESPN--21
● taskESPN--22
● taskESPN--24
● taskESPN--25
● taskESPN--26
● taskESPN--28
● taskESPN--29
● taskESPN--30
● taskESPN--31
● taskESPN--32
● taskESPN--33
● taskESPN--34
● taskESPN--35
● taskESPN--36
● taskESPN--37
● taskESPN--38
● taskESPN--39
● taskESPN--41
● taskESPN--42
● taskESPN--43
● taskESPN--6
● taskESPN--7
● taskESPN--8
ERROR - Task failed
对应任务：
● taskApple--35
● taskApple--36
● taskApple--37
● taskApple--38
● taskESPN--14
● taskESPN--15
● taskESPN--19
● taskESPN--21
错误根因分析
下面这部分不是简单复述报错文本，而是结合代码和日志，对“更像是哪一层出了问题”做归因。
1. responses API 500 Internal Server Error
更可能的原因：
● 这是模型服务端内部错误，不是请求格式错误。
● 当前日志里只出现了 1 次，且不是持续性故障，更像后端短时抖动。
归因层级：
● API 服务端
不是：
● iAgent 沙箱问题
● 浏览器 MCP 问题
● 本地 runner 的参数解析问题
2. MCP 404 Not Found
更可能的原因：
● 本质是浏览器 MCP endpoint 可连通，但某个 MCP 路由或会话对应的资源已经失效。
● 常见场景是 sandbox/browser session 被重置，或者 mcp endpoint 后端状态变化，导致原先可用的工具路由不存在。
归因层级：
● iAgent 浏览器 MCP 层
更具体地说：
● 不是页面内容问题
● 更像 mcpEndpoint 对应的浏览器会话失效、后端路由失配、或 session 生命周期异常
3. MCP 504 Gateway Time-out
更可能的原因：
● 网关已经收到请求，但上游浏览器服务在超时时间内没返回。
● 说明请求已经进入 iAgent 浏览器链路，但具体工具执行卡住了。
归因层级：
● iAgent 浏览器 MCP 层
常见触发点：
● 页面过重
● ESPN 这类动态页面加载慢
● 截图 / DOM 抽取 / navigation 过程中浏览器线程卡住
它通常会进一步引出：
● Tool TimeoutError
● Tool ExceptionGroup
● Task failed
4. Sandbox 429
更可能的原因：
● 平台侧对 sandbox/info 或相关请求做了限流。
● 这不是页面打不开，而是 control plane / metadata 查询被拒。
归因层级：
● iAgent 沙箱控制层
具体表现：
● 获取沙箱信息失败
● Client error : 429
说明：
● 浏览器会话本身不一定已经死掉
● 但因为 BrowserSandbox 每次工具调用前都会先 check_status -> info()，一旦 info() 被限流，后续工具也就无法继续
5. 工具 TimeoutError
更可能的原因：
● 工具调用已经发出，但在 runner / SDK 设定的时间窗口内没拿到完成结果。
● 这通常是底层 MCP 调用的直接表现，而不是上层逻辑错误。
归因层级：
● iAgent 浏览器 MCP 层
常见上游根因：
● 504
● 页面加载过慢
● 截图/观察工具响应超时
6. 工具 ExceptionGroup
更可能的原因：
● 这是 Python anyio/httpx/mcp 异步链路抛出来的包装异常。
● 它本身不是原始根因，而是“底层异步任务组里至少有一个子任务失败”。
归因层级：
● iAgent SDK / MCP 客户端层
通常它后面真正对应的是：
● TimeoutError
● TLS/连接错误
● MCP gateway 失败
所以要把它理解为：
● 二级异常包装
● 不是独立的一种根因
7. 工具 McpError: Session terminated
更可能的原因：
● 浏览器 MCP 会话已经被关闭或中断。
● 后续再对这个 session 调工具时，就会收到 “Session terminated”。
归因层级：
● iAgent 浏览器会话层
更具体地说：
● 不是任务 prompt 的问题
● 是 sandbox 还在，但 browser session 本身已经断开或被重建
8. 工具 type 触发 TypeError: int() argument ... NoneType
更可能的原因：
● 这是我们这层 agent/tool 参数的问题，不是沙箱故障。
● run_iagent_toolcall.py 在执行 type 时，会强制取 parameters["index"] 并转成 int(...)。
● 如果模型输出的 type tool call 缺了 index，或者传了 null，就会直接触发这个错误。
归因层级：
● agent tool 参数层
代码位置：
● run_iagent_toolcall.py
这类问题的本质：
● 不是 iAgent 挂了
● 是模型生成了不合法的 tool 参数，而我们当前没有做更严格的 schema 校验/回退
9. Navigation timeout, but page might still be usable
更可能的原因：
● 这是 iAgent 浏览器工具对 navigate 的一种“软失败”返回。
● 它的语义不是“导航一定失败”，而是“导航在设定时间内没有确认完成，但页面可能已经部分可用”。
归因层级：
● 页面访问层 / 浏览器导航层
这类问题多见于：
● 站点跳转链长
● 前端资源多
● ESPN/Coursera 这类动态页面
● 页面存在地区跳转、脚本重定向、懒加载
它不是最硬的错误，但会显著增加后续失败概率。
10. ERROR - Task failed
更可能的原因：
● 这是我们 runner 的顶层兜底标记，不是根因本身。
● run_iagent_toolcall.py 在 run_task() 里对整个任务做了外层 try/except。
● 只要有未被局部吞掉的异常逃逸出来，就会记一条 ERROR - Task failed。
归因层级：
● runner 顶层异常标记
代码位置：
● run_iagent_toolcall.py
这一类要继续往下看 traceback 才能知道真正原因。当前样本里主要有两种：
1. 沙箱状态非 RUNNING
更可能的根因：
  ○ taskApple--35
  ○ taskApple--36
  ○ taskApple--37
  ○ taskApple--38
  ○ sandbox 创建出来了，但在真正 browser_navigate() 前就已经不处于 RUNNING
  ○ 说明问题在 沙箱控制层/生命周期管理
2. TimeoutError / MCP 连接异常
更可能的根因：
  ○ taskESPN--19
  ○ taskESPN--21
  ○ taskESPN--15
  ○ 浏览器 MCP 连接或导航超时
  ○ 说明问题在 浏览器 MCP 层
从根因角度看，主要问题集中在哪里
按“真正更像的来源”归纳，这批问题主要集中在下面几层：
A. iAgent 浏览器 MCP 层不稳定
典型表现：
● 404 Not Found
● 504 Gateway Time-out
● TimeoutError
● ExceptionGroup
● Session terminated
这是这批里最主要的一类问题。
B. 页面导航层对动态站点适配不足
典型表现：
● Navigation timeout, but page might still be usable
这类问题在 ESPN 上最集中，说明：
● 不是单纯沙箱挂掉
● 更像浏览器对重页面/动态站点的稳定性不足
C. 沙箱控制层偶发异常
典型表现：
● 429
● 沙箱状态非 RUNNING
数量不大，但一旦发生就会直接阻断整个任务。
D. agent 参数层问题
典型表现：
● type 的 index=None
这不是 iAgent 平台问题，而是当前 tool-call runner 对模型输出参数的防御不够。

[Agentic RL] Thinking with image 调研
0. 总结
近半年 think-with-image 的 agentic RL 工作已经从“让模型学会裁图再答题”扩展到五类更复杂的训练范式：
1. 冷启动轨迹几乎成为标配：直接 RL 往往学不到稳定工具使用，或出现格式投机、只调用一次假工具、过度裁剪等问题。因此 DeepEyesV2、V-Thinker、GeoVista、ZoomEarth、MARS、GeoEyes、InSight-o3 等都先用 SFT/trajectory distillation 建立工具调用语法和基本策略，再进入 RL。
2. 奖励从稀疏答案奖励走向过程奖励：ToolsRL 用工具监督先学 zoom/rotate/draw；ZoomEarth/GeoEyes 用 IoU、region-guided、chain-of-focus；MAPO 用 CLIP 语义一致性约束“说到的区域”和“实际看的区域”；InSight-o3 用 in-loop/out-of-loop 混合 bbox 奖励；CodeV/CodeVision 给工具使用正确性、非投机性、IoU 改善等稠密信号。
3. 数据构造从闭源模型蒸馏转向可验证生成：早期主要让 GPT-4o/Gemini 生成轨迹，再筛正确答案；近半年开始大量使用程序化构图、合成 chart/map/document、GT bbox、旋转/翻转标签、OCR/布局检测、retrieval label、沙箱执行结果来减少不可验证轨迹。
4. 工具空间明显扩张：从单一 crop/zoom，扩展到 web search、image search、code execution、rotate/flip、draw line/point、OCR/document retrieval、视觉分割/深度/机器人 grasp/place 等。工具越多，训练越依赖分阶段 curriculum。
5. 领域化很强：遥感、地理定位、chart/map、document/OCR、空间机器人等都在构造专门数据和奖励。通用 agentic RL 方法开始被领域 reward 和领域 benchmark 牵引。
一句话概括：当前最有效路线不是“把工具接上然后 RL”，而是先构造可执行、可验证、难度适中的工具轨迹，让模型具备基本 tool prior，再用分阶段或带过程监督的 RL 校正何时看、看哪里、看完如何答。
1. 工作总览
论文	时间	核心工具/动作	数据构造重点	后训练方法	备注
DeepEyes	2025-05	zoom/crop 等视觉工具	从 HR-Bench、V*Bench、TallyQA、Geometry3K、Visual7W 等构造工具轨迹	SFT + GRPO 风格 RL	用户给定基线，早于半年但影响后续工作
OpenThinkIMG	2025-05	interleaved text-image thoughts	OpenThinkIMG-110K，合成图像中间思维	指令微调	更偏数据范式，RL 色彩较弱
DeepEyesV2	2025-11	code execution、image search、text search	perception/reasoning/search 三类数据，工具可解性过滤，长 CoT 冷启动	SFT + DAPO	从单视觉工具升级到代码+搜索的通用 agent
V-Thinker	2025-11	代码驱动视觉工具、辅助构造	Data Evolution Flywheel，V-Interaction-400K、V-Perception-40K	progressive SFT + GRPO	以“从零合成可执行视觉交互数据”为重点
GeoVista	2025-11	zoom-in、web search	GeoBench + 2K cold-start + 12K RL geolocalization 数据	SFT + GRPO	地理定位，奖励按国家/州/城市分层
ZoomEarth	2025-11	UHR remote-sensing crop/zoom	LRS-GRO，3.5K train/10K test，GPT-4o 轨迹+人工校正	SFT + GRPO	遥感超高分辨率主动感知
CodeV	2025-11	Python code sandbox，图像操作和数学计算	Thyme-SFT 333K + ThinkLite/DeepEyes RL 数据过滤	SFT + TAPO	重点是 tool-aware reward，约束无效/投机代码
CodeVision	2025-12	code-as-tool，无固定工具注册表	约 5K SFT 编程视觉轨迹 + 约 40K RL 样本	SFT + GRPO	OCR/doc/chart 场景下的 crop/rotate/flip/error-handling
ARM-Thinker	2025-12	crop/zoom、validator、doc retrieval	LLaVA-Critic + DeepEyes/MM-IFEngine/MP-DocVQA 等偏好数据	SFT + two-stage GRPO	训练 agentic reward model，不是普通 answer policy
SpaceTools	2025-12	segmentation、depth、pointing、robot grasp/place、code executor	spatial/robotics SFT traces + direct VQA + mock robot API examples	Double Interactive RL	空间/机器人相邻方向，展示大工具空间需分阶段
InSight-o3	2025-12	vSearcher crop 工具	O3-Bench + in-loop collage hard tasks + out-of-loop infographic bbox	Hybrid RL	训练搜索子代理，reasoner/searcher 解耦
SenseNova-MARS	2025-12	text search、image search、crop	约 3K cold-start + FVQA/DeepEyes/Visual-Probe RL 数据	SFT + BN-GSPO	多工具多轮 web/search agent
GeoEyes	2026-02	zoom-in	UHR-CoZ 25,467，多轮 chain-of-zoom	SFT + AdaZoom-GRPO	遥感 UHR，奖励显式刻画是否需要 zoom
NV-CoT	2026-02	连续 bbox action	Visual-CoT bbox 数据 + DeepEyes-style RL 数据	SFT + 连续动作 GRPO	把文本坐标动作改成连续分布采样
MAPO	2026-04	visual action + semantic label	主要复用 DRIM/现有数据	MAPO reward shaping	解决“说在看”和“实际看”不一致
ToolsRL	2026-04	zoom、rotate/flip、draw line/point	DocVQA/V*/HR-Bench/ChartQA/ArxivQA + synthetic chart GT 工具监督	two-stage GRPO	用户给定重点，先工具监督再答案 RL
2. 范式基线
2.1 DeepEyes
论文：DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning
链接：https://arxiv.org/abs/2505.14362
定位
DeepEyes 是后续大量工作的直接基线。它把多模态模型从“只在文本 CoT 中思考”推进到“推理时主动查看图像局部”。典型动作是继续文本推理、调用视觉工具取得局部观察、最终回答。后续 DeepEyesV2、NV-CoT、MARS、GeoEyes 等都显式或隐式沿用其“tool call + observation + answer”的交互格式。
数据构造
● 数据源覆盖高分辨率和细粒度视觉推理：HR-Bench、V*Bench、TallyQA、Geometry3K、Visual7W 等。
● 任务类型大致分为 fine-grained perception、global perception、mathematical reasoning。
● 用闭源/强多模态模型生成带工具调用的 reasoning trajectory。
● 轨迹格式包含 <think>、<tool_call>、<observation>、<answer> 等结构。
● 通过在线 rollout feedback 或答案检查筛掉不正确轨迹，只保留可以得到正确答案、格式可解析、工具调用能执行的样本作为 SFT 冷启动。
后训练方法
● backbone 以 Qwen2.5-VL-7B 为代表。
● 训练分两步：SFT 学会工具调用协议和基本 zoom/crop 策略；RL 在环境中 rollout，工具调用产生新的视觉观察。
● RL 使用 GRPO 风格的 critic-free policy optimization。
● 奖励主要由 answer accuracy 和 format correctness 组成，后续工作常把它视作稀疏 outcome reward 的代表。
● 训练中要过滤全对/全错 prompt group，因为这类 group 没有有效 advantage。
关键经验
● 单纯答案奖励可以让模型在一部分视觉细节任务上学会主动看图，但对复杂工具空间、web search、code execution、domain-specific active perception 不够稳定。
● 后续工作基本都在补 DeepEyes 的三个短板：冷启动数据更可验证、过程奖励更稠密、工具空间更结构化。
2.2 OpenThinkIMG
论文：OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement / Interleaved Visual Reasoning
链接：https://arxiv.org/abs/2505.08617
定位
OpenThinkIMG 更像是“图像作为中间思维”的数据范式基线，而不是典型 agentic RL 系统。它关注让模型产生 interleaved text-image thoughts：推理过程中不只写文字，还生成或引用中间视觉表示。
数据构造
● 构造 OpenThinkIMG-110K。
● 使用 Img-Think 数据构造流程：先得到 caption-enhanced reasoning path，再把部分文本步骤配对到可视化中间图像。
● 使用 completions-as-demonstrations：用强闭源 MLLM 的输出作为示范轨迹。
● 数据目标是让模型学习“何时需要中间视觉状态、如何把视觉中间状态和文本推理对齐”。
后训练方法
● 主要是 instruction tuning / SFT。
● 强化学习不是核心贡献，但它为后续 think-with-image RL 提供了一个关键方向：视觉中间状态本身可以成为 supervision，而不只是工具返回的 observation。
3. 近半年主线工作
3.1 DeepEyesV2
论文：Constructing Agentic Multimodal Models via Tool-Augmented Reasoning and RL
链接：https://arxiv.org/abs/2511.05271
定位
DeepEyesV2 把 DeepEyes 的单一视觉查看扩展为更通用的 agentic multimodal reasoning：模型可以在同一动态推理循环里调用代码执行、图片搜索、文本搜索。它的核心问题不是“能不能裁图”，而是“模型能否在复杂任务中选择合适工具并组合工具”。
工具设计
● code execution：沙箱执行 Python，返回图像、数值测量、日志、错误信息等。
● image search：通过 SerpAPI/Google Lens 风格接口返回 top-k 视觉匹配网页，包含缩略图、标题等。
● text search：返回 top-k 网页标题和 snippet。
● 工具 observation 被追加到上下文，模型继续推理。
● 附录里将工具分为 crop/image operation、math、mark、other、image search、text search 等类别。
数据构造
DeepEyesV2 的数据原则有四个：
● 任务和图像分布要多样，避免只学会某一类工具。
● 结果要可验证，答案格式要结构化。
● 难度要适中，不能全靠 base model 直接答，也不能完全不可解。
● 任务确实要受益于工具，而不是形式上调用工具。
数据分三类：
● perception：细粒度视觉、局部文字、图像区域理解等。
● reasoning：数学、图表、几何、需要代码辅助计算的多步问题。
● search：需要外部知识、图像搜索或文本搜索的问题。
构造流程：
1. 用 Qwen2.5-VL-7B base 对候选问题采样 8 次。
2. 保留 base 直接答对次数不超过 2 次的困难样本，排除太容易的问题。
3. 再允许工具调用采样 8 次，根据工具条件下成功率区分样本。
4. 工具可解样本进入 RL pool。
5. 更难但可通过强模型生成长 CoT 的样本进入 cold-start SFT。
6. 长 CoT 轨迹由 Gemini 2.5 Pro、GPT-4o 等强模型辅助生成，并经过格式/答案筛选。
后训练方法
● backbone：Qwen2.5-VL-7B。
● SFT：batch size 128，learning rate 1e-5，3 epochs。
● RL：使用 DAPO；batch size 256；每个 prompt 16 rollouts；KL 系数 0.0；最大 response length 16384；learning rate 1e-6；clip lower/upper 分别约 0.20/0.30。
● 奖励以 accuracy reward 和 format reward 为主。
● 训练后工具调用平均次数下降，但方差仍存在，说明模型不是简单多调工具，而是在部分任务上形成了更有选择性的工具策略。
关键观察
● 论文报告的 pioneer experiment 很重要：直接对 Qwen2.5-VL 做 RL，模型会写 bug code 后直接答；加 tool bonus 后会 reward hacking，只输出一个不可执行或占位工具调用。这说明多工具 agentic RL 很难从零开始。
● DeepEyesV2 的经验是：冷启动轨迹负责建立 tool prior，RL 负责在真实 rollout 中压缩无效工具调用并提升任务成功率。
3.2 V-Thinker
论文：V-Thinker: Learning to Think with Images via Visual Tool Interaction
链接：https://arxiv.org/abs/2511.04460
定位
V-Thinker 的核心贡献是从零构造大规模可执行视觉交互数据，而不是只从现有 VQA 数据蒸馏轨迹。它提出 Data Evolution Flywheel 和 Visual Progressive Training Curriculum，使模型逐步学会生成/调用视觉工具、观察中间图像、继续推理。
数据构造
主要数据集：
● V-Interaction-400K：视觉交互推理数据。
● V-Perception-40K：视觉感知数据，用于先建立基础视觉理解。
Data Evolution Flywheel 包含几个环节：
1. 知识系统初始化：从 We-Math 2.0 中抽取 1,819 个知识原则。
2. 视觉工具系统初始化和扩展：初始 61 个视觉工具，通过 BGE 聚类、GPT-4.1 和人工归一化扩展到 234 个视觉工具。
3. 样本生成：每个样本包含问题、原始图像生成代码、预测需要的工具、可执行代码片段、每一步视觉状态以及最终答案。
4. 校验器：检查最终答案、原始图像渲染、每个中间视觉状态是否与工具调用一致。
5. 修复器：如果答案错但视觉组件有效，可以重构问题或修复轨迹。
6. 渐进扩展：用 parallel/sequential extension 扩展视觉推理深度，最大深度为 3。
感知数据构造按视觉空间建模：
● surface perception：直接观察元素、数量、颜色、位置。
● semantic reasoning：需要理解对象关系或语义。
● integrated reasoning：结合知识、几何或多步视觉转换。
RL 数据来源：
● We-Math 2.0、MMK12、ThinkLite、V-Interaction-400K。
● 额外加入约 3,000 个 targeted samples：base model 在原图答错，但在编辑或辅助图上答对，用来强化“看图变换确实有用”的场景。
后训练方法
● backbone：Qwen2.5-VL-7B。
● 训练流程：
  a. 先用 V-Perception-40K 做 perception SFT。
  b. 再用 V-Interaction-400K 做 interactive cold-start SFT。
  c. 最后用 GRPO 做 RL，工具执行由 Thyme sandbox 支持。
● 奖励：
R = R_acc + lambda1 * R_format + lambda2 * I[R_acc > 0] * R_tool
● lambda1 = 0.5，lambda2 = 0.3。
● 工具奖励只在最终答案正确时给，避免模型为了工具奖励乱调用工具。
● SFT learning rate 1e-5；RL learning rate 5e-7；RL 一轮，8 rollouts/iteration，warmup 0.05。
关键观察
● V-Thinker 的价值在于把“视觉工具”本身作为可演化系统，而不是固定 crop 工具。
● 它的数据构造比 DeepEyes 系列更程序化，适合需要大量几何、数学、辅助构造图的场景。
3.3 GeoVista
论文：GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization
链接：https://arxiv.org/abs/2511.15705
定位
GeoVista 聚焦图像地理定位。普通 VLM 对地理定位常常缺乏外部知识和局部证据，因此 GeoVista 把 zoom-in 和 web search 接入同一个推理过程，让模型先定位图像中的线索，再检索验证。
数据构造
Benchmark：
● GeoBench：512 张普通照片、512 张 panorama、108 张 satellite images。
● 覆盖 6 大洲、66 个国家、108 个城市。
● 过滤非 localizable 图片和过于 iconic/easy 的 landmark。
● 标签包括 country、province/state、city、lat/lon。
● 评估包含层级准确率和 haversine distance。
训练数据：
● raw panorama 来自 Mapillary API。
● satellite 图像来自 Sentinel-2 / Microsoft Planetary Computer。
● cold-start 数据约 2,000 条。
● RL 数据约 12,000 条。
cold-start 轨迹构造：
1. 强视觉模型提出候选区域、bbox、web-search query 和 rationale。
2. 轨迹中包含先看局部、再搜索、再整合判断的多步 reasoning。
3. 与很多 QA 数据不同，GeoVista 的 cold-start 更偏“策略先验”，不强依赖答案过滤。
后训练方法
● backbone：Qwen2.5-VL-7B-Instruct。
● SFT：约 2K cold-start examples，1 epoch，learning rate 1e-5，global batch 32，max context 32768。
● RL：GRPO via verl，12K samples，global batch 64，mini-batch 32，learning rate 1e-6，无 KL regularization，最多 6 turns，最大上下文 32K。
● 工具 worker 并发执行 zoom 和 web search。
奖励设计：
● 使用分层地理奖励。
● city 正确得分高于 state/province，state/province 高于 country。
● 这种 dense hierarchical reward 比单纯 final exact match 更适合地理定位，因为模型可能国家对但城市错。
关键观察
● GeoVista 是“视觉局部证据 + 外部 web evidence”结合的典型案例。
● 数据和奖励都围绕 geolocation 标签层级设计，不追求通用工具奖励。
3.4 ZoomEarth
论文：ZoomEarth: Active Perception via Multimodal Chain-of-Thought for Ultra-High-Resolution Remote Sensing
链接：https://arxiv.org/abs/2511.12267
定位
ZoomEarth 处理 UHR remote sensing 图像。遥感图像分辨率高、对象小、全局视野和局部细节都重要，普通缩放到固定分辨率会丢失大量信息。ZoomEarth 因此训练模型主动裁剪 ROI 并在裁剪后继续推理。
数据构造
核心数据集：
● LRS-GRO：包含 global、region、object-level QA。
● train 约 3,500，test 约 10,000。
● 评估 active perception quality，使用 APO/IoU 等指标衡量裁剪是否真正找到相关区域。
SFT 数据：
● 第一阶段 SFT 数据中，tool-necessary 与 tool-unnecessary 子集按 2:1 配比。
● GPT-4o 生成两阶段 reasoning-cropping-reasoning CoT。
● 6 名标注者每人超过 10 小时，半自动校正 hallucination、平衡标签。
● global/regional/object 各层级都有精确 bbox 标注。
RL 数据：
● 约 2,500 条样本。
● 评估还包括 MME-RealWorld-RS、XLRS-Bench、GeoLLaVA-8K 等遥感 benchmark。
后训练方法
● backbone：Qwen2.5-VL 3B。
● 输入先缩放到 512，工具可以从原始 UHR 图像裁剪 ROI。
● SFT learning rate 3e-5。
● RL learning rate 1e-7，使用 GRPO。
● RL sampling temperature 0.7，eval temperature 0.01。
● group size G=4，KL gamma 0.04，pattern reward beta 0.05，epsilon 0.2。
奖励设计：
● IoU reward：裁剪 bbox 与目标区域重叠。
● Region-Guided reward：考虑遥感图像的区域层级结构，不只看 box overlap。
● pattern/format reward：约束输出协议。
关键观察
● 论文的消融显示 SFT alone 可能学到格式，但工具使用不一定提升答案；RL 是让工具真正有用的关键。
● 删除 region-guided reward 比删除普通 IoU reward 影响更大，说明领域结构 reward 对遥感 active perception 很重要。
3.5 CodeV
论文：CodeV: Code with Images for Faithful Visual Reasoning
链接：https://arxiv.org/abs/2511.19661
定位
CodeV 让模型在视觉推理中写代码并执行。它不是把工具限制为几个固定 API，而是允许模型用 Python 对图像做 crop、rotation、contrast、measurement，也可以做数学计算。核心问题是代码工具容易 reward hacking：模型可能写无效代码、过大裁剪、重复工具或与问题无关的操作。
数据构造
SFT 数据：
● 来自 Thyme-SFT，从超过 4M raw multimodal examples 中筛选。
● 选择 single-round 和 second-round tool-use examples。
● 总规模约 333K。
● 轨迹包含可执行代码、图像操作、数学计算、sandbox output 和最终答案。
RL 数据：
● 从 ThinkLite-70K 和 DeepEyes-47K 等来源整理。
● 去除外部知识类问题，例如 OK-VQA 类型。
● 用 Qwen2.5-VL-32B 和人工验证清洗噪声标签。
● 对 Qwen2.5-VL-7B 采样 8 次，剔除 empirical accuracy > 0.9 的过易样本，避免 RL 没有有效 advantage。
后训练方法
● SFT 后进入 Tool-Aware Policy Optimization（TAPO），GRPO 风格。
● prompt 明确包含 <think>、<code>、<answer>、<sandbox_output>。
● sandbox 执行 Python，并把图像/文本/数值输出写回上下文。
奖励：
R = lambda_acc * r_acc + lambda_tool * r_tool
● answer reward 包括 exact match、程序化规则和 LLM-as-judge。
● tool reward 检查 sandbox output 是否真的提供与问题相关的证据。
● judge 使用 Qwen2.5-VL-32B，也做过 GPT-5-nano 等替代消融。
● tool reward 特别惩罚 lazy crop、huge crop、invalid/no-op/repeated tool use。
● 工具奖励权重低于答案奖励，避免模型为了工具奖励而牺牲最终答案。
训练设置：
● batch 256，8 rollouts，约 200 steps，learning rate 1e-6，temperature 1.0。
● 训练使用 8 张 H200 GPU。
关键观察
● CodeV 的贡献不只是“允许写代码”，而是把代码输出是否有用做成奖励的一部分。
● 它和 DeepEyesV2 都使用 code execution，但 CodeV 更强调图像处理代码的 faithful evidence，DeepEyesV2 更强调多工具选择和 web/search。
3.6 CodeVision
论文：CodeVision / Thinking with Programming Vision
链接：https://arxiv.org/abs/2512.03746
定位
CodeVision 进一步推动 code-as-tool：不预设固定工具 registry，而是让模型通过代码完成 crop、rotate、flip、error handling、OCR/table/chart 处理。重点场景是文档、OCR、表格、图表和需要图像预处理的问题。
数据构造
SFT 数据：
● 来源包括 handwriting、in-the-wild OCR/VQA、table/chart、math 等数据集。
● 每条样本带 metadata target type：
  ○ single-tool
  ○ multi-tool
  ○ multi-crop
  ○ error-handling
  ○ no-tool
● crop 数据挑选文字区域极小的样本，例如 text region 小于图像面积 0.01%。
● multi-crop 强制连续窗口逐步缩小。
● error-handling 数据包含错误工具调用、runtime error 以及修正后的代码。
● 使用 rotate90/180/270、horizontal/vertical flip 等 metadata-conditioned transforms。
● GPT-5 生成多轮 reasoning/tool action。
● 受控 runtime 将工具输出与 canonical reference 比较，丢弃或修正无效轨迹。
● SFT 约 5,000 条高质量样本。
RL 数据：
● 来自 DocVQA、ThinkLite、TextVQA 等。
● 过滤 all-correct 和 all-incorrect 样本。
● 每条样本标注 must-use tool：rotate、flip、crop 或 None。
● crop 样本带目标 bbox。
● 约 40,000 条 RL training items。
后训练方法
● SFT 阶段只 mask assistant tokens。
● SFT 不进行 online tool execution，而使用缓存 tool outputs。
● RL 使用 GRPO，2 epochs，batch 64，8 rollouts，learning rate 1e-6。
奖励包含三部分：
1. outcome reward：答案正确性和格式。
2. strategy shaping：
  ○ must-use tool reward。
  ○ crop 的 IoU improvement。
  ○ exact sequence bonus。
  ○ 对 optional/suggested tool，根据 rollout comparison 判断使用工具是否带来收益。
3. constraint penalties：
  ○ turn limit。
  ○ inappropriate tool use。
  ○ reward hacking guardrails。
backbone 包括 Qwen2.5-VL-7B、Qwen3-VL-8B/32B。
关键观察
● CodeVision 的数据构造特别强调“错误恢复”，这在真实 code-tool agent 中很重要。
● 它和 ToolsRL 都不满足于 final answer reward，而是显式告诉模型哪类问题应该用什么视觉操作。
3.7 ARM-Thinker
论文：ARM-Thinker: Learning Agentic Multimodal Reward Models via Reinforcement Learning
链接：https://arxiv.org/abs/2512.05111
定位
ARM-Thinker 不是普通解题 policy，而是 agentic multimodal reward model。它训练 reward model 在评价多模态回答时也能主动调用工具，例如裁图、检查指令、检索文档页，从而更准确地判断回答质量。
数据构造
基础偏好数据：
● 从 LLaVA-Critic 等 multimodal preference 数据出发。
新增 agentic task 数据：
● DeepEyes：用于 image crop/zoom-in。
● MM-IFEngine：用于 instruction-following check。
● MP-DocVQA：用于 document retrieval。
负样本构造：
● 使用 GPT-4o-mini 生成 flawed responses。
● 过滤过于相似的 preference pair。
● 难度过滤：如果 base model 5 次 rollout 全部正确，则去掉。
● 用更强 LVLM 生成 CoT 和显式工具轨迹。
● 过滤格式错误、答案错误、工具调用失败的轨迹。
评测：
● ARMBench-VL，1,499 cases，覆盖 3 类任务，支持 single-response RM 和 pairwise RM。
后训练方法
● backbone：Qwen2.5-VL-7B。
● 先 SFT/cold-start，再 two-stage GRPO。
Stage 1：鼓励工具调用。
R_tool = R_f + R_try * I[tool_calls > 0]
Stage 2：精炼准确性和成功工具使用。
● 如果答案错但调用了工具，给 format + try reward。
● 如果答案对但没有成功工具，给 format + answer reward。
● 如果答案对且工具成功，额外给 R_succ。
关键观察
● ARM-Thinker 提醒我们：agentic RL 不只训练 answer policy，也可以训练 judge/reward model。
● 对 agentic evaluation harness 来说，这类方法很相关，因为 evaluator 自己也可能需要看图、检索和执行工具。
3.8 SpaceTools
论文：SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive Reinforcement Learning
链接：https://arxiv.org/abs/2512.04069
定位
SpaceTools 处理空间推理和机器人操作，属于 think-with-image 的相邻但重要方向。它把工具空间扩大到 segmentation、depth、pointing、3D bbox、object pose、grasp/place 等，展示了大工具空间下直接 RL 的困难。
工具与数据构造
Toolshed 包含：
● SAM2 segmentation。
● Depth Pro depth / point cloud。
● RoboRefer / Molmo pointing detectors。
● 3D bbox / object pose。
● GraspGen。
● code executor。
● real/mock robot tools：capture_image、get_depth、execute_grasp、place_object。
训练数据：
● Phase-1 IRL direct VQA。
● Phase-1 SFT teaching tool-use data。
● Phase-2 IRL direct VQA。
● SFT 数据包含 single-tool IRL traces 和 multi-tool demos。
● 机器人 SFT 数据约 500 条 mock robot API examples，使用 HOPE 和 Claude Sonnet 4.5 辅助生成。
● direct VQA 特别平衡 yes/no bias。
后训练方法
Double Interactive RL：
1. Phase 1 用 pointing tool 做 interactive RL，训练一个 IRL teacher。
2. SFT 使用 universal teacher 和 IRL teacher 的轨迹。
3. Phase 2 使用完整工具集做 interactive RL。
训练实现：
● Toolshed + VERL/GRPO。
● rollout 时真实执行工具调用，而不是离线模拟。
● rewards 归一化到 [0, 1]。
● pointing reward 使用 NNDC，比普通距离奖励更稳定。
● KL 约束非常关键。
关键观察
● 论文报告 direct all-tools IRL 容易失败，因为 action space 太大。
● 分阶段训练是大工具空间 agentic RL 的通用经验。
● 结果显示 Tool IRL 在 RoboSpatial 和 unseen RefSpatial 上明显提升，真实机器人 pick/place 也有较高成功率。
3.9 InSight-o3
论文：InSight-o3: Visual Search Agent for High-Resolution Charts and Maps
链接：https://arxiv.org/abs/2512.18745
定位
InSight-o3 把系统拆成 vReasoner 和 vSearcher。vReasoner 负责高层推理和发出搜索请求，vSearcher 负责在高分辨率 chart/map 中找局部区域并 crop。训练目标主要是 vSearcher，而不是整个大模型端到端。
Benchmark 构造
O3-Bench：
● 204 张图片：117 charts，87 maps。
● 345 个 QA：163 chart questions，182 map questions。
● 六选一问题。
● 图像具有高分辨率、高信息密度、多跳定位需求。
数据来源与标注：
● chart 来自 MME-RealWorld Diagram/Table 和互联网。
● 只保留 layout detection 中至少 8 个布局区域的图。
● map 为人工收集的 venue-level maps。
● 使用 PP-DocLayout_plus-L 做 layout pre-annotation。
● Qwen2.5-VL-32B 生成 caption/OCR。
● GPT-5 根据局部 region 生成问题，而不是只看全图。
● 人工过滤和改写。
● 用 GPT-5-mini、Gemini-2.5-Flash、Doubao-Seed-1.6 做难度过滤。
训练数据构造
in-loop RL 数据：
● 从 Visual CoT 和 V* training data 过滤得到带 QA 和 target bbox 的样本。
● 把图像和 distractor 拼成 collage canvas。
● 过滤掉 vReasoner 不用 vSearcher 也能解的问题。
● 得到 15,303 个 hard problems。
out-of-loop RL 数据：
● 来自 InfographicVQA。
● 用 PP-DocLayout boxes 提取区域。
● 去掉 header/footer，保留 text/image/table/chart 等区域。
● GPT-5-nano 生成 region description。
● 构造 (image, region description, bbox) 训练对。
后训练方法
● 两个代理：
  ○ vReasoner：GPT-5-mini 或 Gemini 等强模型。
  ○ vSearcher：Qwen2.5-VL-7B，训练对象。
● vSearcher 使用 crop tool。
Hybrid RL：
● out-of-loop：静态 region localization 任务，有 GT bbox。
● in-loop：vReasoner 在真实答题过程中动态请求 vSearcher，reward 由最终答案和 vReasoner 判断帮助性给出。
奖励：
● out-of-loop：
I[n_tool > 0] * (lambda_format * r_format + lambda_IoU * r_IoU)
● in-loop：用 vReasoner 的 helpfulness 形成 pseudo-IoU，同时要求最终 answer correct。
● out-of-loop 使用 GRPO group normalization。
● in-loop 使用 global mean/std normalization。
超参：
● 最多 6 次工具调用。
● batch 24，8 rollouts，learning rate 1e-6，KL 0.01。
● lambda_format 0.2，lambda_IoU 0.8，IoU threshold alpha 0.25。
● in-loop/out-of-loop 比例 1:1。
● 训练约 150 steps。
● freeze vision tower/adapter。
关键观察
● InSight-o3 的系统解耦很实用：当 reasoning model 很强但局部搜索弱时，只训练 searcher 可以降低成本。
● 它把“看哪里”从最终答案中剥离出来，单独构造 bbox reward。
3.10 SenseNova-MARS
论文：SenseNova-MARS: Multimodal Agentic Reasoning with Search
链接：https://arxiv.org/abs/2512.24330
定位
MARS 是典型多工具多轮 agent：text search、image search、crop 同时存在。任务不是单纯图像细节，而是图像、外部知识和检索证据共同决定答案。
工具设计
● text search：
  ○ RL 训练中使用本地 Wikipedia，例如 enwiki 20250901。
  ○ 使用 E5 retriever 返回 top-5，再用 Qwen3-32B summarizer 压缩。
  ○ inference 时可接 live Serper。
● image search：
  ○ 返回 top-5 titles/thumbnails。
  ○ 训练中预取和缓存。
● crop：
  ○ bbox 使用 [0, 1] 归一化坐标。
数据构造
cold-start SFT：
● 约 3K 到 3.3K 样本。
● FVQA 中筛出约 1,115 条高质量轨迹。
● Pixel-Reasoner warm-start corpus 过滤约 2,000 条。
● curated expert data 约 200 条。
● pipeline：合并 FVQA train、Pixel-Reasoner warm-start、expert QA；Qwen2.5-VL-7B 采样 8 次，正确次数不超过 1 的视为 hard；Gemini-2.5-Flash 合成工具轨迹；GPT-4o 验证格式、逻辑和答案合理性。
RL 数据：
● FVQA 剩余 3,695。
● DeepEyes-4K 约 4,000。
● Visual-Probe 5,729。
● 合计按附录数据约 13.4K。论文不同位置可能有更大规模表述，应以具体数据源表为准。
Benchmark：
● HR-MMSearch：305 张 4K 图像。
● 图片来自 Reuters/AP/CNBC 等新闻源，覆盖 2025 年 8 个领域。
● 问题关注小物体、小文字或需要 search/crop 的细节。
● 根据 Qwen2.5-VL-7B pass@8 分成 hard/easy。
后训练方法
● 7B 模型基于 Qwen2.5-VL。
● SFT：3 epochs，learning rate 1e-5，冻结 vision encoder/projector。
● RL：BN-GSPO via veRL；global batch 128；learning rate 1e-6；KL 1e-4；DAPO-style clip low/high 约 0.2/0.28；最多 10 turns；单 turn 8192 tokens；trajectory 32768 tokens。
● 8B/32B Qwen3-VL 系列主要做 RL。
奖励：
● sequence-level answer accuracy。
● format compliance。
● accuracy 通常由 LLM-as-judge 评估。
● format 约束 <think>、tool call JSON、每个非最终 turn 只能一个工具。
● BN-GSPO 将 GSPO 的序列级 ratio 与 group standardization、batch normalization advantage 结合。
关键观察
● MARS 代表了 think-with-image 与 web/search agent 融合的方向。
● 训练难点在于工具 observation 长、检索噪声大、answer reward 稀疏，因此它依赖较强的 cold-start 和严格格式约束。
3.11 ToolsRL
论文：ToolsRL: Reward is All Tool Learning Needs
链接：https://arxiv.org/abs/2604.19945
定位
ToolsRL 是用户给定的重点论文。它反对只靠 SFT imitation 或 joint answer+tool reward 的训练方式，提出两阶段 RL curriculum：第一阶段只学工具，第二阶段只学任务答案。它的结论很明确：先用工具监督塑造 policy，再用答案奖励优化任务能力，比单阶段混合奖励更稳。
工具设计
ToolsRL 选择 native、interpretable、不依赖外部大模型的工具：
● image_zoom_in_tool
● image_rotate_tool
● image_flip_tool
● image_draw_horizontal_line_tool
● image_draw_vertical_line_tool
● image_mark_points_tool
工具覆盖三类视觉需求：
● zoom：细粒度目标/区域查看。
● rotate/flip：文档方向和镜像纠正。
● draw：chart 上读值、比较、计数时进行视觉辅助标记。
数据构造
训练数据覆盖多个任务源：
● document understanding：3K DocVQA，并进行 rotation/flip augmentation。
● spatial reasoning：V* 和 HR-Bench 4K/8K training sets。
● chart/table：2K ChartQA + 2K ArxivQA。
● synthetic Read-Value：2K。
● synthetic Compare-and-Count：4K。
Stage 1 使用工具监督数据；Stage 2 使用统一 QA 数据。ChartQA/ArxivQA 因缺少工具标注，不进入 Stage 1，但用于 Stage 2。
不同工具的数据标注方式：
● zoom-in：
  ○ 利用 object/region annotations 得到 GT bbox。
  ○ 任务目标是让模型裁到能覆盖目标证据的区域。
● rotate/flip：
  ○ 对 document images 做旋转/翻转增强。
  ○ GT 是正确 orientation transform。
● draw：
  ○ 合成 chart-style read-value 和 compare-and-count 任务。
  ○ 生成 scatter/line charts，通常有 8-20 个点，标签打乱。
  ○ 保留完整坐标 metadata。
  ○ read-value 有 reference point 和要读的点。
  ○ compare/count 预计算满足条件的点以及应画的 horizontal/vertical lines 或 mark points。
后训练方法
backbone：
● Qwen2.5-VL-7B-Instruct。
优化：
● GRPO。
● 论文设置中每个 prompt 采样多条 trajectories，工具监督和答案监督分阶段进行。
Stage 1：Tool Supervision
● 只优化工具特定奖励，不给 answer accuracy reward。
● prompt 中只开放当前任务需要的工具，降低动作空间。
● 目标是先让模型形成稳定工具行为。
工具奖励：
1. Zoom reward
  ○ 使用 Modified F1 / ModF1 衡量 pixel overlap。
  ○ false positive 权重较低，false negative 权重较高，例如 w_fp=0.1, w_fn=1。
  ○ 这鼓励 crop 覆盖目标区域，宁可略大也不要漏掉证据。
2. Rotate/Flip reward
  ○ 正确 orientation transform 得 1，否则 0。
3. Draw reward
  ○ 用统一 coordinate reward。
  ○ 通过 Hungarian matching 匹配预测线/点和 GT 线/点。
  ○ 对 line/point 使用连续相似度，而不是只做 exact match。
Stage 1 总奖励：
R_stage1 = 0.5 * (R_global_tool + R_answer_tool) + R_format
● R_global_tool 看整条轨迹中是否出现正确工具行为。
● R_answer_tool 看最终 answer image 上的工具行为是否正确。
● global-only 会过度探索；answer-only 又太保守；二者结合效果最好。
Stage 2：Task Accuracy
● 使用统一 QA prompt，开放完整 toolbox。
● 只使用 answer accuracy + format reward，不再给 tool-specific supervision。
● accuracy 对真实 QA 多用 LLM judge，对 synthetic chart 可用 normalized numeric score。
关键观察
● 单阶段 joint tool+accuracy reward 容易失败，模型会退回 text-only。
● Stage 1 学到的工具策略能在 Stage 2 保留下来，即便 Stage 2 没有显式 tool bonus。
● 训练后模型会在需要时多次使用工具，约 3-5 calls/sample，而不是只做一次形式化调用。
● ToolsRL 对后续工作的启发是：工具学习可以先脱离最终 QA，像学动作技能一样训练。
3.12 MAPO
论文：MAPO / Multimodal Action-aware Policy Optimization for MLLM Tool Use
链接：https://arxiv.org/abs/2604.06777
定位
MAPO 聚焦一个常见但容易被忽视的问题：模型文本上说“我在看这个区域”，但实际工具 action 裁到的图像可能不是那个区域。也就是 reasoning-action discrepancy。MAPO 通过要求模型给每次视觉 action 附带 descriptive label，并用视觉语义相似度来约束 action 和文字意图。
数据构造
● MAPO 本身不是大规模新数据集工作，主要复用现有 think-with-images 数据，例如 DRIM 相关开源数据。
● 重点在训练目标设计，而不是新建 benchmark。
● 在训练格式上，要求每次工具 action 都包含对目标视觉内容的自然语言描述。
后训练方法
核心机制：
1. 模型输出 action 和 action label。
2. 环境执行 action，返回 visual observation。
3. 使用 CLIP 计算 label 与返回图像之间的 semantic similarity。
4. 将这个相似度作为过程奖励，与最终 outcome reward 结合。
奖励/advantage 特点：
● trajectory-aware discount，lambda 约 0.95。
● beta 约 0.4，用于结合 semantic score 和 outcome reward。
● 用 group advantage estimation 做 policy update。
● 最多 6 个 interaction turns。
● 强制 label 和长度约束，减少 reward hacking。
● 实现基于 verl。
关键观察
● MAPO 不追求更多工具，而是让工具动作和语言推理对齐。
● 对所有 crop/zoom 类方法都很有借鉴价值：如果只奖励 answer，模型可能偶然答对；如果只奖励 bbox，模型可能不解释为什么裁这里。MAPO 把“看哪里”和“为什么看”绑定。
3.13 NV-CoT
论文：NV-CoT: Towards Non-verbal Chain-of-Thought for MLLMs
链接：https://arxiv.org/abs/2602.23959
定位
NV-CoT 关注坐标动作的表达方式。很多 think-with-image 方法把 bbox 写成文本 token，例如 [x1, y1, x2, y2]，这会引入离散 tokenization 误差，也不适合 RL 中连续动作建模。NV-CoT 改为直接预测连续 bbox action。
数据构造
SFT 设置：
● 使用 Visual-CoT 数据。
● Visual-CoT 来自 12 个公开数据集，覆盖 text/document understanding、fine-grained recognition、VQA、chart understanding、relation reasoning。
● text/document 任务的 bbox 来自 OCR 工具，例如 PaddleOCR。
● 其他任务利用 part/attribute boxes、object/scene graph annotations。
● 数据源包括 TextVQA、DocVQA、DUDE、SROIE、InfographicsVQA、Flickr30k、Visual7W、VSR、GQA 等。
● 每条样本包含 image、question、中间 bbox grounding steps、final answer。
RL 设置：
● 使用与 DeepEyes 类似的数据源：
  ○ V* train 的 fine-grained visual search。
  ○ ArxivQA 的 chart understanding。
  ○ ThinkLite-VL 的 general reasoning。
● 通过多次 response/answer consistency 过滤 trivial 和 extremely hard 样本。
● 将问题改写为 open-ended。
● 移除不可验证、歧义和 unreadable 样本。
● RL 数据不需要中间 bbox annotations。
后训练方法
模型结构：
● 在语言模型上增加 coordinate heads，分别预测 x1、y1、x2、y2 和 scale。
● 坐标分布使用 Gaussian 或 Laplace。
● SFT loss = text token CE + coordinate regression loss。
● Gaussian 对应 L2，Laplace 对应 L1 风格目标。
RL：
● 扩展 GRPO 到连续动作。
● 使用 reparameterized sampling。
● 对连续 bbox action 计算 analytic importance ratio 和 KL。
● inference 时使用均值坐标。
● reward 继承 DeepEyes 风格：answer correctness + format validity + conditional zoom-in bonus。
训练设置：
● SFT 一轮，batch 128。
● RL 在 Qwen2.5-VL-7B 上，80 iterations，32 张 A100，batch 256，group size 16。
关键观察
● NV-CoT 的贡献是动作表示层面的：把“文本化坐标”变成真正的连续视觉 action。
● 这对需要高精度 bbox 的 active perception 很重要，也能减少 token-level 坐标生成的偶然性。
3.14 GeoEyes
论文：GeoEyes: Adaptive Chain-of-Zoom for Remote Sensing Vision-Language Models
链接：https://arxiv.org/abs/2602.14201
定位
GeoEyes 是遥感 UHR 场景的 chain-of-zoom agent。它比 ZoomEarth 更强调多轮 zoom 的策略：什么时候不 zoom，什么时候单次 zoom，什么时候逐步 coarse-to-fine zoom。
数据构造
核心数据：
● UHR-CoZ：Ultra-High-Resolution Chain-of-Zoom 数据。
● 从 HighRS-VQA 转换而来。
● 使用 agent pipeline 生成多轮 focusing annotations。
● 每条轨迹经过 format validation、execution validation 和 interaction validation。
数据规模和统计：
● 25,467 samples。
● 平均图像尺寸约 2178 x 2051。
● 平均问题长度约 103.2 tokens。
● 平均 reasoning 长度约 157.8 tokens。
● 93.6% 样本使用工具。
● 平均工具调用 1.1 次。
● 深度分布：
  ○ D=1，无 zoom：6.4%。
  ○ D=2，一次 zoom：86.7%。
  ○ D>=3，多轮 progressive zoom：6.9%。
任务类型：
● object classification。
● color。
● spatial relation。
● counting。
● route planning。
● anomaly detection。
● 其他 remote-sensing specific reasoning。
RL 数据：
● 在 DeepEyes baseline 数据之外加入 SuperRS-VQA，提高遥感任务覆盖。
后训练方法
训练流程：
1. cold-start SFT on UHR-CoZ。
2. AdaZoom-GRPO。
工具：
● 只使用 zoom_in。
● bbox 归一化到 1000 x 1000 坐标系。
奖励：
format-gated weighted combination：
● R_acc：答案正确性。
● R_format：格式。
● R_tool：adaptive efficiency。
● R_cof：Chain-of-Focus。
● R_proc：process verification。
Adaptive efficiency：
● 按任务类别设置免费工具使用 quota。
● 根据 base model 对样本的置信/难度调节惩罚，难样本允许更多 zoom。
● 难度项类似 P_alpha = 1 - p(y|x)_base。
Chain-of-Focus：
● 奖励 coarse-to-fine containment。
● 允许中性 backtracking。
● 惩罚视野漂移。
● 使用 directional IoU / containment，而不是普通对称 IoU。
Process Verification：
● necessity-aware judge 判断问题是否需要 zoom。
● 如果细节问题不 zoom 但直接自信回答，会被惩罚。
关键观察
● SFT ablation 显示：无 SFT 时模型可能工具调用率 100% 但 accuracy 低；UHR-CoZ 冷启动后工具调用更适度、accuracy 更高。
● GeoEyes 的核心是“自适应”，不是简单鼓励多 zoom。
3.15 DRIM / Deep But Reliable
论文：Deep But Reliable: Interleaved Thinking with Images in MLLMs
链接：https://arxiv.org/abs/2601.02783
定位
DRIM 是近半年 interleaved thinking-with-images 的重要相关工作。它强调 detail-awareness 和 logical consistency，并提出 benchmark、数据和多模态 RL 方案。由于本次重点放在用户给定的 DeepEyes/ToolsRL 及直接 agentic tool-use RL，这里作为补充。
数据构造
● 构造 DRIM-Bench。
● 构造约 84K curated multimodal reasoning instruction dataset。
● 数据形式强调 interleaved textual reasoning 和 visual perception feedback。
后训练方法
● 使用 multimodal reinforcement learning。
● reward 混合 visual perception feedback 和 textual reasoning validation。
● 方向上与 MAPO、NV-CoT 类似，都是把“视觉过程是否可靠”纳入 RL，而不只看最终答案。
4. 横向比较
4.1 数据构造范式
闭源模型轨迹蒸馏
● 代表：DeepEyes、DeepEyesV2、GeoVista、MARS、ZoomEarth。
● 优点：快，能得到自然语言 CoT 和工具调用。
● 缺点：轨迹质量依赖 teacher；工具调用可能不最优；容易把 teacher hallucination 蒸馏进学生。
● 常见补救：答案过滤、格式过滤、工具执行过滤、人工校正、base model 难度过滤。
程序化/可验证合成
● 代表：V-Thinker、ToolsRL synthetic charts、CodeVision transforms。
● 优点：有 GT bbox、orientation、line/point、intermediate state，过程奖励可信。
● 缺点：分布可能偏合成，迁移到真实图像需要混合真实数据。
领域 benchmark 反向构造训练数据
● 代表：GeoVista、ZoomEarth、GeoEyes、InSight-o3。
● 优点：数据和 reward 与领域难点强绑定。
● 缺点：泛化到其他领域不一定好，工程成本高。
从“工具必要性”筛样
● 代表：DeepEyesV2、MARS、CodeV、InSight-o3、GeoEyes。
● 常见做法：
  ○ base model 多次采样，去掉直接能答对的样本。
  ○ 工具模型多次采样，保留工具可解样本。
  ○ 去掉 all-correct/all-wrong prompt group。
● 目的：让 RL group 内有 advantage 差异，并确保工具确实有用。
4.2 后训练范式
SFT + sparse outcome RL
● 代表：DeepEyes、DeepEyesV2、MARS。
● 简洁，但对工具行为的信用分配弱。
● 适合工具空间较小或冷启动很强的场景。
SFT + process/tool reward RL
● 代表：ZoomEarth、GeoEyes、CodeV、CodeVision、InSight-o3。
● 用 IoU、region-guided、tool relevance、pseudo-IoU、strategy shaping 等补足最终答案稀疏性。
● 对 active perception 更有效。
两阶段工具技能 RL
● 代表：ToolsRL。
● Stage 1 只学工具动作；Stage 2 只学 QA。
● 避免 joint reward 下模型回到 text-only 或工具投机。
大工具空间 curriculum
● 代表：SpaceTools。
● 先单工具/小工具空间，再多工具。
● 对机器人和复杂工具箱尤其重要。
动作表示改造
● 代表：NV-CoT。
● 将 bbox 从文本 token 改成连续 action distribution。
● 更适合高精度视觉动作和连续控制式 RL。
语言-动作一致性优化
● 代表：MAPO。
● 强制每个 action 有 semantic label，用 CLIP 等视觉相似度约束 action 与意图。
● 解决“推理文本合理但实际工具看错地方”的问题。
4.3 奖励设计谱系
奖励类型	代表工作	解决的问题	风险
answer accuracy	DeepEyes、DeepEyesV2、MARS	直接优化最终任务	稀疏，难以信用分配
format reward	几乎所有工作	保证工具调用可解析	容易被格式投机
tool bonus	V-Thinker、DeepEyes 变体	鼓励使用工具	若不 gated，会乱调工具
IoU / bbox reward	ZoomEarth、ToolsRL、InSight-o3、GeoEyes	指导看哪里	需要 GT bbox 或 pseudo bbox
region/domain reward	ZoomEarth、GeoVista、GeoEyes	利用领域层级	迁移性弱
code/tool usefulness reward	CodeV、CodeVision	防止 no-op、huge crop、无关代码	judge 成本高
semantic action consistency	MAPO	对齐“说在看”和“实际看”	CLIP 相似度可能被语义捷径影响
continuous action KL/ratio	NV-CoT	支持连续 bbox RL	实现复杂，需稳定分布建模
stage-specific reward	ToolsRL、SpaceTools	降低动作空间和训练冲突	pipeline 更复杂
4.4 主要失效模式
1. 工具格式投机：模型输出看似合法的 tool call，但参数无效、不可执行或没有实际观察价值。
2. text-only collapse：如果答案奖励更容易通过语言先验获得，模型会放弃工具。
3. over-tool-use：加粗暴 tool bonus 后，模型每题都调用工具，甚至多次重复调用。
4. huge crop / lazy crop：crop 覆盖整图或大区域，形式上调用工具但没有聚焦。
5. action-reasoning mismatch：文本说要看 A，实际裁到 B。
6. search noise amplification：web/image search 返回噪声，模型把检索片段当事实。
7. all-correct/all-wrong group：RL group advantage 失效，训练不稳定。
8. 工具空间过大：直接 RL 在多工具环境中探索成本太高。
4.5 目前最稳的 recipe
如果要复现或构建新的 think-with-image agentic RL 系统，当前最稳路线是：
1. 先定义少量 native tools，保证每个工具 deterministic、可记录、可复放。
2. 构造工具必要性数据：base model 直接答不稳，但工具后可解。
3. 为每种工具准备至少一种可验证过程标签：
  ○ crop/zoom：bbox 或 IoU。
  ○ rotate/flip：orientation transform。
  ○ draw/mark：line/point coordinates。
  ○ code：执行输出与 evidence relevance。
  ○ search：retrieval evidence 与答案引用。
4. 做 cold-start SFT，让模型学会协议、何时停止、如何读取 observation。
5. RL 初期限制工具空间或做工具技能阶段。
6. RL 后期开放完整工具箱，用 answer reward + format + gated process reward。
7. 每轮 rollout 记录：
  ○ 原图。
  ○ tool call 参数。
  ○ tool observation。
  ○ final answer。
  ○ reward breakdown。
  ○ judge 解释或规则命中。
8. 单独评估工具行为，不只看 final accuracy：
  ○ tool call rate。
  ○ avg calls/sample。
  ○ valid call rate。
  ○ useful call rate。
  ○ bbox IoU / containment。
  ○ answer with-tool vs without-tool gap。
5. 对 agentic eval / harness 的启发
对于本仓库这类 agentic evaluation harness，最值得吸收的是评估字段和可复放日志，而不是直接照搬某一个 RL 算法：
1. 评估时保留完整 trajectory：仅保存 final answer 不够，至少要保存 tool call、tool observation、judge reward breakdown。
2. 把工具成功和答案成功分开统计：例如答案对但工具无效、工具正确但答案错，是两种不同 failure。
3. 设计 tool-necessity 子集：有些题不该用工具，强行鼓励工具会降低鲁棒性。
4. 加入过程指标：
  ○ crop 是否覆盖 GT 或 evidence region。
  ○ search result 是否被正确引用。
  ○ code 输出是否被最终答案使用。
5. 做 difficulty filtering：评估集也可按 base pass@k 分层，否则工具 agent 的收益会被过易样本稀释。
6. 防止 reward hacking：如果未来 harness 接入训练回路，需要显式检查 no-op tool、huge crop、重复调用、格式占位符。
6. 论文索引
缩写	标题	链接
DeepEyes	DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning	https://arxiv.org/abs/2505.14362
OpenThinkIMG	OpenThinkIMG: Learning to Think with Images	https://arxiv.org/abs/2505.08617
DeepEyesV2	Constructing Agentic Multimodal Models via Tool-Augmented Reasoning and RL	https://arxiv.org/abs/2511.05271
V-Thinker	V-Thinker: Learning to Think with Images via Visual Tool Interaction	https://arxiv.org/abs/2511.04460
GeoVista	GeoVista: Web-Augmented Agentic Visual Reasoning for Geolocalization	https://arxiv.org/abs/2511.15705
ZoomEarth	ZoomEarth: Active Perception for Ultra-High-Resolution Remote Sensing	https://arxiv.org/abs/2511.12267
CodeV	CodeV: Code with Images for Faithful Visual Reasoning	https://arxiv.org/abs/2511.19661
CodeVision	CodeVision / Thinking with Programming Vision	https://arxiv.org/abs/2512.03746
ARM-Thinker	ARM-Thinker: Agentic Multimodal Reward Models	https://arxiv.org/abs/2512.05111
SpaceTools	Tool-Augmented Spatial Reasoning via Double Interactive RL	https://arxiv.org/abs/2512.04069
InSight-o3	InSight-o3: Visual Search Agent for High-Resolution Charts and Maps	https://arxiv.org/abs/2512.18745
SenseNova-MARS	SenseNova-MARS: Multimodal Agentic Reasoning with Search	https://arxiv.org/abs/2512.24330
DRIM	Deep But Reliable: Interleaved Thinking with Images in MLLMs	https://arxiv.org/abs/2601.02783
GeoEyes	GeoEyes: Adaptive Chain-of-Zoom for Remote Sensing VLMs	https://arxiv.org/abs/2602.14201
NV-CoT	NV-CoT: Towards Non-verbal Chain-of-Thought for MLLMs	https://arxiv.org/abs/2602.23959
MAPO	Multimodal Action-aware Policy Optimization	https://arxiv.org/abs/2604.06777
ToolsRL	ToolsRL: Reward is All Tool Learning Needs	https://arxiv.org/abs/2604.19945
[Office Survey]
下周计划：
1. benchmark的大表合并
2. 分享准备
