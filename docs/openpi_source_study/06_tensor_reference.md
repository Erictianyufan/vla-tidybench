# 06｜张量、调用链与源码位置速查

本章用于复习和查找，不替代前五章的概念解释。

## 1. 统一记号

```text
B  batch size
P  全部Prefix token数（图像 + 文本/状态）
L  tokenized prompt的padding长度，π0.5默认最大200
H  action horizon，TidyBench为16
D  模型动作宽度，TidyBench为32
Da 物理动作维度，TidyBench为7
Ds 原始状态维度，TidyBench为18
```

## 2. TidyBench 张量总表

| 变量 | 语义 | 形状 |
|---|---|---:|
| table/wrist raw image | Isaac/LeRobot 图像 | 各 `[B,200,200,3]` 或单样本无 B |
| model image slot | Resize 后图像 | `[B,224,224,3]` |
| single-view image tokens | SigLIP 输出 | `[B,256,2048]` |
| raw state | 关节位置/速度 | `[B,18]` |
| padded state field | 模型统一字段 | `[B,32]` |
| tokenized prompt | 指令 + 离散18D状态 | `[B,L]` |
| prefix tokens | 图像 + 语言/状态 | `[B,P,2048]` |
| physical action chunk | 项目动作标签 | `[B,16,7]` |
| padded actions | π0.5 动作标签 | `[B,16,32]` |
| noise `ε` | 高斯动作噪声 | `[B,16,32]` |
| flow time `t` | 每个样本一个时间 | `[B]` |
| noised action `x_t` | Flow 路径中间点 | `[B,16,32]` |
| action tokens | Action Expert 输入 | `[B,16,1024]` |
| AdaRMS condition | 时间条件 | `[B,1024]` |
| suffix output | Action Expert 输出 | `[B,16,1024]` |
| target velocity `u_t` | `ε-a` | `[B,16,32]` |
| predicted velocity `v_t` | 模型速度场 | `[B,16,32]` |
| chunked loss | 动作维均值后 | `[B,16]` |
| scalar loss | batch/horizon均值后 | `[]` |
| inference output | 去batch/反归一化/裁剪后 | `[16,7]` |

## 3. 完整推理调用链

```text
Policy.infer(obs)
  -> input transforms
       data mapping
       Normalize
       ResizeImages
       TokenizePrompt(discrete state)
       PadStatesAndActions
  -> add batch / move device
  -> Observation.from_dict
  -> Pi0.sample_actions
       preprocess_observation
       embed_prefix
       llm([prefix,None]) -> KV cache
       x_1 ~ N(0,I)
       while t >= 0:
         embed_suffix(x_t,t)
         llm([None,suffix], kv_cache, adarms_cond)
         action_out_proj -> v_t
         x_t <- x_t + dt*v_t
  -> remove batch / NumPy
  -> Unnormalize
  -> LiberoOutputs裁到7D
  -> [16,7]
```

## 4. 完整训练调用链

```text
scripts/train.py::main
  -> create_data_loader
  -> init_train_state
       create model
       load pi05-DROID params
       plan FSDP sharding
       init optimizer state
  -> jit(train_step)
  -> loop
       train_step
         nnx.merge(model_def,params)
         compute_loss
           preprocess_observation(train=True)
           sample ε,t
           x_t=tε+(1-t)a
           u_t=ε-a
           embed_prefix + embed_suffix
           dual-expert forward
           action_out_proj -> v_t
           mean((v_t-u_t)^2, action_dim)
         mean -> scalar loss
         value_and_grad
         optimizer.update
         apply_updates
         update TrainState
       logging
       next batch
       periodic Orbax save
```

## 5. 每层双 Expert 调用链

```text
Prefix x_p [B,P,2048]              Action x_a [B,16,1024]
  -> regular RMSNorm                 -> AdaRMS(x_a,time)
  -> own Q/K/V projection            -> own Q/K/V projection
                 \                  /
                  concat on token axis
                          -> RoPE
                          -> QK^T / sqrt(head_dim)
                          -> attention mask
                          -> softmax * V
                  split on token axis
                 /                   \
  own output proj -> 2048             own output proj -> 1024
  residual + own FFN                  gated residual + own FFN
```

## 6. OpenPI 源码索引

适用于 commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`。

| 文件/函数 | 阅读问题 |
|---|---|
| `src/openpi/models/model.py::Observation` | 标准 Observation 和 Actions 形状是什么？ |
| `src/openpi/models/model.py::preprocess_observation` | 图像 resize/增强和 image mask 如何处理？ |
| `src/openpi/models/pi0_config.py::Pi0Config` | `pi05=True`、action width/horizon、离散状态如何配置？ |
| `src/openpi/models/tokenizer.py::PaligemmaTokenizer` | 指令和归一化状态如何组成 token？ |
| `src/openpi/transforms.py` | Normalize、Tokenize、Padding 如何实现？ |
| `src/openpi/policies/policy.py::Policy.infer` | 外部字典如何进入模型、输出怎样回到 NumPy？ |
| `src/openpi/policies/policy_config.py::create_trained_policy` | transforms、checkpoint 和 Policy 如何组装？ |
| `src/openpi/models/pi0.py::make_attn_mask` | AR 边界怎样变成二维注意力矩阵？ |
| `src/openpi/models/pi0.py::embed_prefix` | 图像、文本和状态 token 如何拼接？ |
| `src/openpi/models/pi0.py::embed_suffix` | 动作和时间如何进入 Action Expert？ |
| `src/openpi/models/pi0.py::compute_loss` | Flow Matching 训练样本和监督怎样构造？ |
| `src/openpi/models/pi0.py::sample_actions` | KV Cache 和反向 Euler 积分怎样生成动作？ |
| `src/openpi/models/gemma.py::RMSNorm` | 普通 RMSNorm 和 AdaRMS 的分支差异是什么？ |
| `src/openpi/models/gemma.py::Attention` | 两个 Expert 的 Q/K/V 在哪里合并和拆分？ |
| `src/openpi/models/gemma.py::Block` | Attention/FFN 的调制和门控残差顺序是什么？ |
| `src/openpi/training/data_loader.py::create_data_loader` | RLDS/LeRobot 数据怎样选择和 batch？ |
| `src/openpi/training/config.py::ModelTransformFactory` | π0.5 model transforms 的准确顺序是什么？ |
| `scripts/train.py::init_train_state` | 权重加载、参数过滤和 sharding 如何初始化？ |
| `scripts/train.py::train_step` | Loss、梯度、AdamW 和 TrainState 如何更新？ |
| `scripts/train.py::main` | 完整实验生命周期如何调度？ |
| `src/openpi/training/sharding.py` | Data Parallel 与 FSDP 如何布局？ |
| `src/openpi/training/checkpoints.py` | params、optimizer state 和 assets 如何保存？ |

## 7. 项目侧源码索引

| 文件 | 与 OpenPI 的连接 |
|---|---|
| [`stack_config.py`](../../source/vla_tidybench/openpi/stack_config.py) | 锁定 π0.5 LoRA、32D、16 horizon、checkpoint和训练参数 |
| [`drawer_config.py`](../../source/vla_tidybench/openpi/drawer_config.py) | 将同一结构切换到 drawer 多技能数据集 |
| [`convert_stack_to_lerobot.py`](../../scripts/convert_stack_to_lerobot.py) | 把 Isaac episode 转为 OpenPI 可读的 LeRobot 字段 |
| [`compute_drawer_norm_stats.py`](../../scripts/compute_drawer_norm_stats.py) | 复用固定 OpenPI 计算归一化资产 |
| [`train_drawer_pi05.py`](../../scripts/train_drawer_pi05.py) | 将项目配置传入外部 OpenPI `train.py` |
| [`smoke_drawer_policy.py`](../../scripts/smoke_drawer_policy.py) | 从 Orbax checkpoint 恢复并执行真实观测离线推理 |
| [`action_adapter.py`](../../source/vla_tidybench/policy_bridge/action_adapter.py) | 7D物理动作与Isaac控制动作的唯一权威转换 |
| [`action_queue.py`](../../source/vla_tidybench/policy_bridge/action_queue.py) | 接收16步动作块并逐步消费 |

## 8. 固定阅读笔记模板

```markdown
## 本次问题

## 调用链
A -> B -> C

## 核心函数
- 谁调用它：
- 输入及形状：
- 输出及形状：
- 改变的是形状还是语义：
- 下一步由谁使用：

## 确认的事实
1.
2.

## 暂时作为黑盒
-

## 未解决问题
-

## 不看源码复述
用5到10句话解释本次主线。
```

## 9. 常见误读

- 把 `Policy.infer()` 当成模型本体；它主要是适配壳。
- 把 AR 边界 `True` 当成“这个 token 被屏蔽”。
- 把动作块内部理解为逐动作自回归；π0.5 是整块联合去噪。
- 把 Flow Matching `v_t` 当成末端物理速度。
- 把时间调制说成“每步修改 Transformer 权重”；实际动态生成激活调制量。
- 把 `action_dim=32` 当成机器人有 32 个控制自由度。
- 只恢复模型参数却忽略 AdamW state，然后声称是严格续训。
- 把两步 smoke checkpoint 当成训练完成的策略。

