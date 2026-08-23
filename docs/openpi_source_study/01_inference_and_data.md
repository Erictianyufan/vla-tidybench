# 01｜推理入口与 TidyBench 数据链

本章只追踪一件事：一条 TidyBench 观测怎样变成 π0.5 输入，并最终返回一个 `16×7` 动作块。

## 1. 项目侧原始语义

TidyBench 采集和部署必须保持同一接口：

| 字段 | 项目语义 | 单帧形状 |
|---|---|---:|
| table camera | 第三人称 RGB，`uint8 HWC` | `[200,200,3]` |
| wrist camera | 腕部 RGB，`uint8 HWC` | `[200,200,3]` |
| state | 9D 关节位置 + 9D 关节速度 | `[18]` |
| action | 6D 相对末端运动 + 1D 夹爪 | `[7]` |
| prompt | 原子技能语言指令 | 字符串 |

项目实现：

- HDF5 读取与 18D 状态：[`isaac_hdf5.py`](../../source/vla_tidybench/data/isaac_hdf5.py)
- 7D 权威动作转换：[`action_adapter.py`](../../source/vla_tidybench/policy_bridge/action_adapter.py)
- LeRobot 转换：[`convert_stack_to_lerobot.py`](../../scripts/convert_stack_to_lerobot.py)

物体位姿、抽屉关节和仿真序列化状态不进入可部署策略输入。

## 2. 训练数据的变换顺序

OpenPI `training/data_loader.py::transform_dataset()` 按以下固定顺序执行：

```text
LeRobot sample
  -> repack_transforms
  -> data_transforms
  -> Normalize
  -> model_transforms
  -> Observation + Actions
```

### 2.1 Repack

`LeRobotLiberoDataConfig` 将数据集键映射为策略键：

```text
image        -> observation/image
wrist_image  -> observation/wrist_image
state        -> observation/state
actions      -> actions
prompt       -> prompt
```

Repack 只在训练数据侧执行。推理环境本来就应发送策略侧字段，因此不需要再模拟数据集内部命名。

### 2.2 Data transforms

`LiberoInputs` 将两路真实相机放入 π0.5 固定的三个图像槽：

```text
base_0_rgb        <- table camera       mask=True
left_wrist_0_rgb  <- wrist camera       mask=True
right_wrist_0_rgb <- 全零占位图          mask=False
```

占位图的数值本身不是重点，`mask=False` 才表示第三路相机不存在。不能用一张重复图并标成有效输入。

### 2.3 Normalize

归一化统计来自训练集 assets。状态和动作先进入统一数值范围，推理必须加载同一份统计量。

顺序必须是“先归一化，再离散状态”，因为离散器的分桶范围按归一化数值设计。如果直接离散原始关节量，不同量纲会集中到错误区间。

### 2.4 Model transforms

π0.5 的 `ModelTransformFactory` 依次执行：

```text
InjectDefaultPrompt
ResizeImages(224,224)
TokenizePrompt(discrete_state_input=True)
PadStatesAndActions(32)
```

这里有一个容易忽略的细节：`TokenizePrompt` 在 padding 之前运行，因此离散进文本的是归一化后的真实 18D 状态；随后保留在 `Observation.state` 字段里的状态才补到 32D。π0.5 的 Action Expert 不再创建连续 state token，主要状态条件已经在 Prefix 文本中。

动作链是：

```text
单步7D物理动作
  -> DataLoader按action_horizon切成[16,7]
  -> Normalize
  -> PadStatesAndActions
  -> [16,32]
```

TidyBench 配置明确设置 `extra_delta_transform=False`，因为采集标签已经是项目定义的 7D IK-relative 动作，不应再次做 delta 转换。

## 3. `Policy.infer()` 做什么

OpenPI 入口：`src/openpi/policies/policy.py::Policy.infer`。

逻辑可以压缩为：

```python
inputs = copy(obs)
inputs = input_transform(inputs)
inputs = add_batch_and_move_to_device(inputs)
observation = Observation.from_dict(inputs)
actions = model.sample_actions(rng_or_device, observation)
outputs = remove_batch_and_to_numpy(actions)
outputs = output_transform(outputs)
return outputs
```

逐步解释：

1. **浅复制输入树**：transform 可能原地改字典，避免污染调用方的观测。
2. **执行输入 transforms**：统一键名、图像、归一化、tokenization 和 padding。
3. **添加 batch 维**：外部单样本 `[H,W,C]` 变成模型需要的 `[1,H,W,C]`。
4. **转换设备数组**：JAX 路径转 `jax.Array`，PyTorch 路径转 `torch.Tensor` 并移动到设备。
5. **构造 `Observation`**：把松散字典变成模型约定的数据结构。
6. **调用 `sample_actions()`**：这一行才进入真正的模型推理。
7. **移除 batch 维并转 NumPy**：`[1,16,32] -> [16,32]`。
8. **输出 transforms**：反归一化，并通过 `LiberoOutputs` 裁回前 7 维。

所以 `Policy.infer()` 不是神经网络本体，而是模型前后的适配壳；真正执行 π0.5 的是 `self._sample_actions(...)`。

## 4. `sample_actions()` 的外层流程

OpenPI 入口：`src/openpi/models/pi0.py::Pi0.sample_actions`。

```text
Observation
  -> preprocess_observation
  -> embed_prefix
  -> Prefix前向一次，保存18层KV Cache
  -> x_1 ~ N(0,I)，形状[B,16,32]
  -> 对每个去噪时间步：
       embed_suffix(x_t,t)
       Action Expert读取Prefix Cache
       预测v_t
       x_t <- x_t + dt*v_t
  -> 返回x_0
```

Prefix 只计算一次，因为同一次动作采样中，图像、指令和状态不变；变化的只有当前噪声动作 `x_t` 和时间 `t`。

## 5. TidyBench 推理维度表

设单样本推理 `B=1`：

| 阶段 | 变量 | 形状 |
|---|---|---:|
| 环境输入 | table/wrist RGB | 各 `[200,200,3]` |
| Resize 后 | 三个图像槽 | 各 `[1,224,224,3]` |
| 原始状态 | `state` | `[18]` |
| 离散状态 | Prefix 文本的一部分 | 18 个离散值对应的 token 序列 |
| Padding 后状态字段 | `Observation.state` | `[1,32]` |
| 模型动作噪声 | `x_1` | `[1,16,32]` |
| 模型输出 | actions | `[1,16,32]` |
| 去 batch | actions | `[16,32]` |
| 反归一化/裁剪后 | 物理动作块 | `[16,7]` |

不要把三种维度混为一谈：

```text
18 = 机器人本体状态维度
7  = 项目物理动作维度
32 = π0.5统一动作宽度
```

## 6. 阅读验收

合上源码后应能回答：

1. 为什么训练有 Repack，而部署不一定有？
2. 为什么归一化必须早于状态离散化？
3. 第三路全零图为什么还必须配 `mask=False`？
4. batch 维在哪里添加、在哪里移除？
5. `[16,7]` 在哪里变成 `[16,32]`，输出又在哪里裁回 7D？
6. `Policy.infer()` 和真正模型执行的边界在哪里？

下一章进入 Prefix/Suffix、块编号、AR mask 和双 Expert Attention。

