# 实验记录：EPMI 诊断流水线

**记录人**：Jinglin Xu  
**最后更新**：2026-06-12

---

# 第二轮：四域全量 + 完整 MCQ 输入（正式版）

**日期**：2026-06-12  
**状态**：已完成  
**SLURM 作业 ID**：19162684

## 1. 本轮变更

相比第一轮，本轮做了以下改动：

| 项目 | 第一轮 | 第二轮 |
|---|---|---|
| 数据域数量 | 1（GSM8K） | 4（GSM8K、medmcqa、ARC、MMLU） |
| 每域样本量 | 512 | 2,048 |
| C4 参考集样本量 | 256 | 1,024 |
| MCQ 输入格式 | 仅题目文本 | 题目 + 完整选项（A/B/C/D） |
| GSM8K 格式 | 纯文本 | 纯文本（无变化，无选项） |

**完整 MCQ 输入的理由**：路由器感知的是输入序列的全部 token，仅喂题目而省略选项会遗漏模型实际在推断时接收的信息结构（标签字母、选项语义），导致路由统计偏低，无法反映真实的部署条件。

## 2. 硬件与环境

| 项目 | 值 |
|---|---|
| 节点 | gpua048.delta.ncsa.illinois.edu |
| GPU | NVIDIA A100-SXM4-40GB |
| 运行时间 | 22:47:35 → 23:10:58（约 23 分钟） |
| Python | 3.11.14 / PyTorch 2.9.1+cu128 / Transformers 4.57.3 / Datasets 5.0.0 |

## 3. 数据集与输入样例

### C4（预训练参考集）
- 来源：`allenai/c4`，`en` 子集，`train` split，streaming=True，seed=0
- 实际处理：**1,024 条 / 287,686 tokens**
- 预训练路由熵：mean H = 4.1436，range [4.1333, 4.1554]（接近理论最大值 ln64 = 4.1589）

### GSM8K — 数学应用题（formatter: `plain`）
- 来源：`openai/gsm8k`，`main` 子集，`train` split
- **实际使用：2,048 条**
- 输入格式：仅题目文本，无答案，无选项
- 输入样例：
  ```
  Stefan goes to a restaurant to eat dinner with his family. They order an appetizer
  that costs $10 and 4 entrees that are $20 each. If they tip 20% of the total for
  the waiter, what is the total amount of money that they spend at the restaurant?
  ```

### medmcqa — 医学执照考试 MCQ（formatter: `medmcqa_mcq`）
- 来源：`medmcqa`，`train` split（无子集）
- **实际使用：2,048 条**（数据集共 182,822 条，按需采样）
- 输入格式：题目 + A/B/C/D 四个选项（字段 opa/opb/opc/opd）
- 输入样例：
  ```
  True statements about asbestosis
  A. Causes Lung Ca
  B. Pleural mesothelioma
  C. Peritoneal mesothelioma
  D. All of the above
  ```

### ARC-Challenge — 小学科学 MCQ（formatter: `arc_mcq`）
- 来源：`allenai/ai2_arc`，`ARC-Challenge` 子集，`test` split
- **实际使用：1,172 条**（test split 仅 1,172 条，低于请求的 2,048，已发出 [warn]）
- 输入格式：题目 + 选项（choices.label 与 choices.text 拼合）
- 输入样例：
  ```
  Cities control the amount of pollution that is allowed to come from cars.
  How does this most likely help people?
  A. The air stays cleaner.
  B. Cars can travel at faster speeds.
  C. The skills of the drivers improve.
  D. It becomes safer to drive on the roads.
  ```

### MMLU — 大学水平学科知识 MCQ（formatter: `mmlu_mcq`）
- 来源：`cais/mmlu`，`all` 子集，`test` split
- **实际使用：2,048 条**（test split 共 14,042 条，随机抽取）
- 输入格式：题目 + A/B/C/D（choices 列表按顺序标注）
- 输入样例：
  ```
  A state built a casino and issued bonds to finance its construction. On five
  occasions, there were episodes of violence in various casinos in the state...
  Is this law likely to be held constitutional if most casinos in the state were
  owned by those from out-of-state?
  A. Yes, because the act was expressly authorized by the state legislature.
  B. Yes, but only if the local interest in safety outweighs the burden of interstate commerce.
  C. No, because out-of-state casinos are part of interstate commerce.
  D. No, because the statute violates the due process rights of the owners of the casinos.
  ```

## 4. 诊断结果

### 4.1 综合 EPMI 汇总

| 域 | n_samples | EPMI\_RED | EPMI\_EAD | **EPMI** |
|---|---|---|---|---|
| gsm8k | 2,048 | 0.0143 | 0.0148 | **0.0145** |
| medmcqa | 2,048 | 0.0485 | 0.0512 | **0.0498** |
| arc | 1,172 | 0.0176 | 0.0239 | **0.0207** |
| mmlu | 2,048 | 0.0024 | 0.0100 | **0.0062** |

**排序**：medmcqa (0.050) ≫ arc (0.021) > gsm8k (0.015) ≫ mmlu (0.006)

### 4.2 逐层 RED / EAD 明细

| 层 | RED_gsm8k | EAD_gsm8k | RED_medmcqa | EAD_medmcqa | RED_arc | EAD_arc | RED_mmlu | EAD_mmlu |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0054 | 0.0049 | 0.0073 | 0.0111 | 0.0060 | 0.0081 | 0.0018 | 0.0043 |
| 1 | 0.0017 | 0.0033 | 0.0045 | 0.0069 | 0.0026 | 0.0041 | 0.0011 | 0.0021 |
| 2 | 0.0044 | 0.0058 | 0.0170 | 0.0212 | 0.0057 | 0.0082 | 0.0020 | 0.0042 |
| 3 | 0.0051 | 0.0084 | 0.0136 | 0.0168 | 0.0060 | 0.0090 | 0.0016 | 0.0047 |
| 4 | 0.0057 | 0.0074 | 0.0146 | 0.0185 | 0.0063 | 0.0097 | 0.0017 | 0.0054 |
| 5 | 0.0076 | 0.0108 | 0.0186 | 0.0271 | 0.0081 | 0.0149 | 0.0013 | 0.0063 |
| 6 | 0.0099 | 0.0156 | 0.0219 | 0.0301 | 0.0076 | 0.0153 | 0.0015 | 0.0081 |
| 7 | 0.0061 | 0.0093 | 0.0131 | 0.0207 | 0.0076 | 0.0160 | 0.0006 | 0.0071 |
| 8 | 0.0118 | 0.0160 | 0.0161 | 0.0207 | 0.0095 | 0.0205 | 0.0002 | 0.0064 |
| 9 | 0.0120 | 0.0198 | 0.0226 | 0.0265 | 0.0069 | 0.0163 | 0.0016 | 0.0085 |
| 10 | 0.0048 | 0.0094 | 0.0270 | 0.0402 | 0.0059 | 0.0171 | 0.0005 | 0.0074 |
| 11 | 0.0135 | 0.0184 | 0.0462 | 0.0629 | 0.0111 | 0.0219 | 0.0011 | 0.0087 |
| 12 | 0.0178 | 0.0235 | 0.0469 | 0.0759 | 0.0251 | 0.0461 | 0.0026 | 0.0140 |
| 13 | 0.0163 | 0.0219 | 0.0469 | 0.0784 | 0.0199 | 0.0352 | 0.0032 | 0.0150 |
| 14 | 0.0093 | 0.0135 | 0.0643 | 0.0941 | 0.0168 | 0.0296 | 0.0025 | 0.0162 |
| 15 | 0.0085 | 0.0116 | 0.0380 | 0.0646 | 0.0150 | 0.0291 | 0.0010 | 0.0115 |

**深层集中模式**：所有域的 EAD 峰值均集中在第 11–14 层，medmcqa 在第 14 层达到最大值 EAD=0.094，是 gsm8k 同层的 7 倍。

### 4.3 稳定性检验（3 次独立重采样）

| 域 | 指标 | Draw 1 | Draw 2 | Draw 3 | 均值 | CV | 结论 |
|---|---|---|---|---|---|---|---|
| gsm8k | RED | 0.0140 | 0.0141 | 0.0139 | 0.0140 | 0.006 | ✓ OK |
| gsm8k | EAD | 0.0145 | 0.0146 | 0.0144 | 0.0145 | 0.007 | ✓ OK |
| medmcqa | RED | 0.0492 | 0.0492 | 0.0495 | 0.0493 | 0.004 | ✓ OK |
| medmcqa | EAD | 0.0520 | 0.0518 | 0.0523 | 0.0520 | 0.004 | ✓ OK |
| arc | RED | 0.0176 | 0.0176 | 0.0176 | 0.0176 | 0.000 | ✓ OK（全量=整个test split）|
| arc | EAD | 0.0239 | 0.0239 | 0.0239 | 0.0239 | 0.000 | ✓ OK（全量=整个test split）|
| mmlu | RED | 0.0029 | 0.0025 | 0.0027 | 0.0027 | 0.078 | ✓ OK |
| mmlu | EAD | 0.0107 | 0.0102 | 0.0103 | 0.0104 | 0.024 | ✓ OK |

所有域 CV < 0.10，全部通过稳定性阈值。

## 5. 分析与解读

### 5.1 域排序的直觉解释

- **medmcqa（0.050）**：高度专业化的医学词汇（解剖名称、疾病、药物）与 C4 网页文本分布差距最大。选项本身（如"Pleural mesothelioma"）也引导路由器激活了预训练时罕见的专业化 expert。
- **arc（0.021）**：小学科学题，语言结构简单，但涉及因果推理，与网页文本有一定结构差异。
- **gsm8k（0.015）**：数学应用题，是流畅英文日常叙述，OLMoE 预训练时见过大量类似文本，路由几乎不需要调整。
- **mmlu（0.006）**：MMLU 覆盖 57 个学科，高度多样性导致路由在所有 expert 上均匀分布，与 C4 的均匀路由高度一致，EPMI 接近于 0。这是**多样性稀释效应**：不是没有错位，而是各学科的错位方向相互抵消。

### 5.2 与第一轮 GSM8K 的对比

| 设置 | EPMI (第一轮) | EPMI (第二轮) |
|---|---|---|
| GSM8K | 0.0149 | 0.0145 |

两轮 GSM8K 结果几乎相同（差 0.0004），说明从 256→1024 条 C4 参考集、以及 512→2048 条 GSM8K 均未改变结论，基准稳健。

### 5.3 MCQ 格式对 medmcqa 的影响

早期仅喂题目时 medmcqa EPMI ≈ 0.032；加入选项后升至 0.050（提升 56%）。选项中的医学术语（如"Peritoneal mesothelioma"）是路由偏移的重要来源，在实际推断场景中模型会接收完整 MCQ，因此使用完整格式更能反映真实错位程度。

## 6. 保存的文件

```
results/run_20260612_224930/
├── config.json          超参 + 各域输入样例（domain_input_examples 字段）
├── pretrain_stats.pt    C4 1024条参考集 RoutingStats [avg_routing_probs, entropy, expert_load]
├── gsm8k_stats.pt       GSM8K 2048条 RoutingStats
├── medmcqa_stats.pt     medmcqa 2048条 RoutingStats
├── arc_stats.pt         ARC 1172条 RoutingStats
├── mmlu_stats.pt        MMLU 2048条 RoutingStats
├── metrics.json         所有标量指标 + 逐层向量 + 稳定性数据
└── summary.txt          stdout 完整输出
```

---

# 第一轮：GSM8K 基线（仅题目文本）

**日期**：2026-06-12  
**记录人**：Jinglin Xu  
**状态**：已完成，由第二轮取代

---

## 1. 背景与目标

### 研究问题

对一个已训练好的 MoE 模型，在做领域微调之前，能否通过纯前向传播（不更新任何参数）诊断出：**现有 expert pool 是否能吸收目标域的监督信号，还是需要先扩充 expert pool？**

### 本轮实验目标

1. 搭建 EPMI 诊断流水线，在 OLMoE-1B-7B 上跑通
2. 以 GSM8K（数学题，预期低错位）为第一个测试域，建立基线
3. 验证指标稳定性（CV < 0.10）
4. 为后续加入医疗域（medmcqa，预期高错位）做对比准备

---

## 2. 实验环境

### 2.1 硬件

| 项目 | 配置 |
|---|---|
| 集群 | NCSA Delta HPC |
| 节点 | gpua037.delta.ncsa.illinois.edu |
| GPU | NVIDIA A100-SXM4-40GB × 1 |
| CPU | 8 cores |
| 内存 | 64 GB |
| SLURM account | bexq-delta-gpu / gpuA100x4 partition |
| 作业 ID | 19152648 |
| 实际运行时间 | 20:42:21 → 20:47:21（约 5 分钟） |

### 2.2 软件

| 包 | 版本 |
|---|---|
| Python | 3.11.14 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 4.57.3 |
| Datasets | 5.0.0 |
| Accelerate | 1.12.0 |
| tqdm | 4.67.1 |
| conda 环境路径 | `/projects/bexq/yxu30/conda/envs/olmoe` |

### 2.3 模型

| 项目 | 值 |
|---|---|
| 模型 ID | `allenai/OLMoE-1B-7B-0924`（base，非 instruct）|
| 精度 | bfloat16 |
| 加载方式 | `device_map="auto"`（单卡自动放置）|
| MoE 层数 | 16（全部 transformer 层均为 MoE） |
| 每层 expert 数 | 64 |
| 每 token 激活 expert 数 | top-8 |
| 总参数 | ~7B（激活参数 ~1B） |
| 模型缓存路径 | `/work/hdd/bexq/yxu30/hf/transformers/` |

---

## 3. 方法与实现

### 3.1 代码结构

```
moe-epmi/
├── config.py              EPMIConfig dataclass，所有超参集中于此
├── data_loader.py         load_texts()，返回 List[str]
├── routing_extractor.py   extract_routing_stats()，核心前向传播
├── metrics.py             compute_red(), compute_ead(), compute_epmi()；RLC stub
├── run_diagnostics.py     主脚本：流程编排、表格输出、结果保存
└── run_epmi.sh            SLURM 作业脚本
```

### 3.2 数据集使用

#### C4（预训练参考集）

- **来源**：`allenai/c4`，`en` 子集，`train` split
- **采样量**：256 条（快速验证阶段；正式实验建议升至 1024）
- **采样方式**：流式读取（streaming=True），shuffle buffer=10,000，seed=0
- **使用字段**：`text`
- **token 长度分布**：min=23，max=11,441，中位数=247，均值=547
  - 超过 512 tokens 的条目：78/256（30%），被截断至 512
  - 截断对路由统计无实质影响（分析见第 4.3 节）
- **实际处理 token 数**：74,049

#### GSM8K（目标域，通用数学）

- **来源**：`openai/gsm8k`，`main` 子集，`train` split
- **采样量**：512 条
- **采样方式**：加载全部 7,473 条后随机打乱（seed=42），取前 512
- **使用字段**：`question`（仅题目文本，不含答案）
- **token 长度分布**：min=10，max=**217**，中位数=52，均值=55.9
  - p99 = 123 tokens；**0 条超过 512 tokens，截断完全未发生**
- **实际处理 token 数**：28,357（512 条 × 平均约 55 tokens）
- **典型样本**：
  > "Natalia sold clips to 48 of her friends in April and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

### 3.3 前向传播流程（每个 batch）

1. **分词**：batch_size=8，padding=True，truncation=True，max_length=512，padding_side="right"
2. **前向传播**：`model(input_ids, attention_mask, output_router_logits=True)`
3. **取路由 logits**：`outputs.router_logits` 为 16 个 tensor 的 tuple，每个形状 `[B×T, 64]`（bfloat16）
4. **过滤 padding**：`flat_mask = attention_mask.view(-1).bool()`，只保留 mask=1 的位置
5. **计算两种路由表示**：
   - `softmax(logits_f)` → 每 token 的 64 维概率分布（用于 RED/EAD）
   - `topk(logits_f, k=8)` → 每 token 实际激活的 8 个 expert（用于 load/RLC）
6. **累加**：对所有 batch 的 token 求和，最后除以总 token 数得到边际分布

### 3.4 指标定义

#### 边际路由分布

对每层 $l$，将所有（非 padding）token 的 softmax 概率向量求平均：

$$P_l(e) = \frac{1}{N} \sum_{t=1}^{N} \text{softmax}(\text{logit}_t)_e \quad \in \mathbb{R}^{64}$$

#### RED — Routing Entropy Drop

$$H_l = -\sum_{e=1}^{64} P_l(e) \log P_l(e)$$

$$\text{RED}_l = \max\!\left(0,\ 1 - \frac{H_l^\text{target}}{H_l^\text{pre}}\right)$$

$$\text{EPMI\_RED} = \text{mean}\!\left(\text{top-}{\lceil L/3 \rceil}\ \{RED_l\}\right) = \text{mean of top-5 layers}$$

- 衡量目标域路由的集中程度相对预训练增加了多少
- RED=0 表示集中程度与预训练相同或更低；RED=1 表示完全坍缩到单个 expert

#### EAD — Expert Activation Divergence

$$\text{JSD}(P \| Q) = \frac{1}{2}\text{KL}(P\|A) + \frac{1}{2}\text{KL}(Q\|A), \quad A = \frac{P+Q}{2}$$

$$\text{EAD}_l = \frac{\text{JSD}(P_l^\text{target} \| P_l^\text{pre})}{\ln 2} \in [0, 1]$$

$$\text{EPMI\_EAD} = \sum_{l=1}^{L} w_l \cdot \text{EAD}_l, \quad w_l = \frac{l/L}{\sum_{l'} l'/L}$$

- 衡量目标域路由的结构是否与预训练不同（不同的 expert 被激活）
- 深层权重更高，因为深层编码更多任务特定信息
- 除以 $\ln 2$ 归一化至 $[0,1]$

#### RLC — Residual Loss Concentration（未实现）

$$\text{RLC}_l = \text{Pearson\_Corr}\left(\text{load}_l,\ \text{mean\_loss\_per\_expert}_l\right)$$

需要同时拿到 per-token CE loss，目前预留接口，待后续实现。

#### 综合 EPMI

$$\text{EPMI} = \frac{\text{EPMI\_RED} + \text{EPMI\_EAD}}{2}$$

---

## 4. 实验结果

### 4.1 预训练参考分布（C4）

| 层 | $H_l^\text{pre}$ |
|---|---|
| 全层均值 | **4.1444** |
| 最大值 | 4.1555（第 1 层） |
| 最小值 | 4.1357（第 15 层） |
| 理论最大值 $\ln 64$ | 4.1589 |

**观察**：C4 下的路由熵极其均匀，所有层均在理论最大值的 99.9% 以上，且层间差异极小（极差 < 0.02）。这表明 OLMoE 在预训练时学到了高度均匀的负载均衡——expert 之间几乎没有专业化偏好。

### 4.2 GSM8K 诊断结果

| 指标 | 值 |
|---|---|
| **EPMI\_RED** | **0.0145** |
| **EPMI\_EAD** | **0.0153** |
| **EPMI（综合）** | **0.0149** |
| 处理 token 数 | 28,357 |

#### 逐层明细

| 层 | RED | EAD | 备注 |
|---|---|---|---|
| 0 | 0.0055 | 0.0050 | |
| 1 | 0.0017 | 0.0033 | 全层最低，第2层路由最稳定 |
| 2 | 0.0045 | 0.0061 | |
| 3 | 0.0053 | 0.0085 | |
| 4 | 0.0060 | 0.0077 | |
| 5 | 0.0077 | 0.0110 | |
| 6 | 0.0101 | 0.0155 | |
| 7 | 0.0065 | 0.0097 | |
| 8 | 0.0120 | 0.0161 | |
| 9 | 0.0120 | 0.0197 | |
| 10 | 0.0047 | 0.0094 | |
| 11 | 0.0137 | 0.0189 | |
| 12 | **0.0176** | **0.0239** | 全层最高 |
| 13 | 0.0169 | 0.0225 | |
| 14 | 0.0100 | 0.0143 | |
| 15 | 0.0092 | 0.0130 | |

EAD 的层间趋势：浅层（0–5）< 中层（6–10）< 深层（11–13）> 尾层（14–15）。第 12–13 层为峰值区。

### 4.3 稳定性检验

三次独立重采样（seed=142, 242, 342），预训练参考分布固定不变：

| 指标 | Draw 1 | Draw 2 | Draw 3 | 均值 | CV |
|---|---|---|---|---|---|
| EPMI\_RED | 0.0140 | 0.0143 | 0.0142 | 0.0142 | **0.008** ✓ |
| EPMI\_EAD | 0.0148 | 0.0151 | 0.0148 | 0.0149 | **0.012** ✓ |

两个指标 CV 均远低于阈值 0.10，说明 512 条样本量对 GSM8K 已足够稳定。

---

## 5. 分析与解读

### 5.1 为什么 GSM8K 的 EPMI 这么低

EPMI ≈ 0.015 意味着：
- 路由熵下降约 1.5%（RED）
- 路由结构偏移约 1.5%（EAD，归一化后）

从直觉上讲，GSM8K 的问题文本（数学应用题）与 C4（网页文本）在语言层面高度重叠——都是流畅的英文句子，只是话题是数学。OLMoE 在预训练时见过大量包含数字和数学表达的网页，因此 expert 路由几乎不需要调整。

### 5.2 关于截断的影响分析

- **GSM8K**：无截断（最长 217 tokens，全部完整输入）。我们的统计覆盖了每道题从第一个 token 到最后一个 token 的完整路由行为。
- **C4**：30% 的文章被截断至 512 tokens。但由于我们用 C4 的目的只是建立路由基准（"预训练时路由是什么样的"），截断不影响结论：同一篇文章的前 512 tokens 和后半段是同类型的网页文本，路由器行为不会系统性偏差。512 tokens 也恰好对应模型训练时每个 step 实际处理的粒度。

### 5.3 预训练路由的高均匀性

C4 下的路由熵（mean=4.1444）接近理论最大值 $\ln 64 = 4.1589$，说明 OLMoE 在预训练时几乎实现了完美的负载均衡。这是 OLMoE 论文本身设计的目标，此处得到验证。

对后续实验的含义：RED 的分母（预训练熵）非常大且稳定，这意味着 RED 对目标域的路由集中非常敏感——即使是轻微的集中（如医疗术语导致某几个 expert 被过度激活），也会被 RED 捕捉到。

### 5.4 层间规律

EAD 在深层（第 11–13 层）更高，与参考文献中"深层编码更多任务特异性信息"的假设一致。这也是 EAD 聚合时给深层更高权重的依据。即使对于 GSM8K 这个"低错位"域，深层的结构偏移仍然比浅层高约 4–5 倍。

---

## 6. 本轮局限与待确认问题

| 问题 | 说明 |
|---|---|
| 预训练参考样本量小 | 目前 256 条，建议升至 1024 再做最终对比 |
| RLC 未实现 | 缺少"过载 expert 是否也是高损失 expert"这一维度 |
| 只有一个域 | 无法判断 0.015 是否"低"——需要 medmcqa 对比 |
| question-only 模式 | 只喂题目，不喂答案，RLC 实现后需决定是否改为 question+answer |
| C4 参考集来自 train split | 若日后改用专门的预训练数据样本，需重新建立基准 |

---

## 7. 保存的文件

```
results/run_20260612_204437/
├── config.json          本次所有超参（model、datasets、metric 设置）
├── pretrain_stats.pt    C4 参考集的完整 RoutingStats tensor bundle
│                        → avg_routing_probs [16, 64]
│                        → entropy [16]
│                        → expert_load [16, 64]
├── gsm8k_stats.pt       GSM8K 的完整 RoutingStats tensor bundle（同上结构）
├── metrics.json         所有标量指标 + 逐层向量 + 稳定性数据（JSON 格式）
└── summary.txt          stdout 完整输出文本
```

加载方式：
```python
import torch
pretrain = torch.load("results/run_20260612_204437/pretrain_stats.pt")
gsm8k    = torch.load("results/run_20260612_204437/gsm8k_stats.pt")
# pretrain["avg_routing_probs"].shape → [16, 64]
```

---

## 8. 后续计划

1. ~~**加入 medmcqa、ARC、MMLU**~~：✅ 已在第二轮完成（SLURM 19162684）
2. ~~**升高预训练参考样本量至 1024**~~：✅ 已在第二轮完成（287,686 tokens）
3. **实现 RLC**：扩展 routing_extractor.py，在前向传播时同时收集 per-token CE loss 和 per-layer top-k 分配，然后补全 metrics.py 中的 compute_rlc()
4. **加更多域做 EPMI 排序**：覆盖代码、法律、金融等，验证 EPMI 是否单调预测微调增益
5. **ECI 实验**：在高 EPMI 域上实施 Expert Creation Intervention，验证 EPMI 的预测有效性

---

*本记录基于 `EPMI_reference.md` 中的方法定义，结合实际实现和运行结果整理。*
