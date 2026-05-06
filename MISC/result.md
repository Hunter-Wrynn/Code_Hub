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
