# 03｜残差连接与 AdaRMS 时间调制

本章先澄清一句容易误解的话：

> 时间 `t` 不会在每次前向时重写 Transformer 的固定权重；它通过固定的、已训练的投影层动态生成 `scale/shift/gate`，调节本次前向的中间激活和残差强度。

## 1. 为什么需要残差连接

普通深层网络让下一层完全替换当前特征：

```text
x_next = F(x)
```

残差网络保留一条恒等路径：

```text
x_next = x + F(x)
```

这一层只需学习“在原有信息上补充什么”。如果暂时没学好，让 `F(x)≈0`，网络仍近似保持 `x_next≈x`，也为反向传播提供了更直接的梯度通道。

数值例子：

```text
x    = [2.0, -1.0]
F(x) = [0.3,  0.4]
x_next = [2.3, -0.6]
```

Transformer Block 有两条残差分支：

```text
x'     = x  + Attention(Norm(x))
x_next = x' + FFN(Norm(x'))
```

## 2. 门控残差

OpenPI `src/openpi/models/gemma.py::_gated_residual`：

```python
if gate is None:
    return x + y
return x + y * gate
```

π0.5 Action Expert 使用：

```text
x_next = x + gate(t) * y
```

其中 `y` 是 Attention 或 FFN 给出的修正量。`gate` 不是一个布尔开关，而是每个隐藏特征都有一个连续值：

- `gate=0`：本次完全保留 `x`；
- `gate=1`：普通残差 `x+y`；
- `0<gate<1`：只注入部分修正；
- 负值或大于 1：可学习的反向/放大修正。

例如：

```text
x    = [2.0, -1.0]
y    = [0.3,  0.4]
gate = [0.2,  0.5]
x + y*gate = [2.06, -0.80]
```

## 3. 时间条件如何产生

入口：`src/openpi/models/pi0.py::Pi0.embed_suffix`。

```text
t：[B]
  -> posemb_sincos(t, 1024)
  -> Linear(1024,1024) + Swish
  -> Linear(1024,1024) + Swish
c_t：[B,1024]
```

正弦余弦编码用多组频率表达 `[0,1]` 内的细粒度位置；MLP 再把通用时间编码变成适合 Action Expert 的条件向量 `c_t`。

推理调用明确传入：

```python
adarms_cond=[None, c_t]
```

因此 PaliGemma Expert 不受 flow time 调制，Action Expert 受调制。图像与任务含义在一次采样中不随去噪时间变化，候选动作的处理策略却必须随 `t` 改变。

## 4. RMSNorm 做什么

OpenPI `src/openpi/models/gemma.py::RMSNorm` 首先计算：

```text
var = mean(x², axis=-1, keepdims=True)
x_hat = x / sqrt(var + eps)
```

即：

\[
\hat{x}=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}
\]

它按每个 token 的隐藏维稳定数值尺度，不减均值，因此与 LayerNorm 不完全相同。

普通 PaliGemma 路径只有一个学习到的静态 scale：

```text
RMSNorm(x) * (1 + learned_scale)
```

## 5. AdaRMS 如何生成三组调制量

Action Expert 的 `cond` 不为 `None` 时：

```python
modulation = Dense(3 * hidden_dim)(cond)
scale, shift, gate = split(modulation[:, None, :], 3)
normed = normed * (1 + scale) + shift
```

TidyBench Action Expert 隐藏宽度为 1024：

```text
c_t：[B,1024]
  -> 当前RMSNorm自己的Dense(1024,3072)
modulation：[B,3072]
  -> split + 添加序列维
scale：[B,1,1024]
shift：[B,1,1024]
gate： [B,1,1024]
```

应用到动作块：

```text
x：[B,16,1024]
scale/shift：[B,1,1024]
  -> 沿16个动作位置广播
```

数学形式：

\[
\operatorname{AdaRMS}(x,t)
=\hat{x}\odot(1+\gamma(t))+\beta(t)
\]

三个量的职责：

| 调制量 | 控制对象 | 直观含义 |
|---|---|---|
| `scale` / `γ` | 归一化特征的幅度 | 哪些隐藏特征应增强或减弱 |
| `shift` / `β` | 特征中心 | 把网络切换到怎样的工作状态 |
| `gate` / `g` | 残差分支输出 | 当前 Attention/FFN 应修改原信息多少 |

## 6. 一个 Block 内如何执行

入口：`src/openpi/models/gemma.py::Block.__call__`。

Attention 子层：

```text
x_l
  -> AdaRMS(x_l,c_t)，得到调制后的输入和gate_attn
  -> Attention
  -> x'_l = x_l + gate_attn * attention_output
```

FFN 子层：

```text
x'_l
  -> 另一个AdaRMS(x'_l,c_t)，得到gate_ffn
  -> FFN
  -> x_(l+1) = x'_l + gate_ffn * ffn_output
```

完整公式：

\[
x'_l=x_l+g_l^{attn}(t)\odot
\operatorname{Attention}
(\operatorname{AdaRMS}_l^{attn}(x_l,t))
\]

\[
x_{l+1}=x'_l+g_l^{ffn}(t)\odot
\operatorname{FFN}_l
(\operatorname{AdaRMS}_l^{ffn}(x'_l,t))
\]

## 7. 为什么同一个时间能让每层行为不同

18 层都接收同一个 `c_t`，但每层 Attention 前和 FFN 前的 AdaRMS 都有自己的投影参数：

```text
第1层：  W_1_attn(c_t), W_1_ffn(c_t)
第2层：  W_2_attn(c_t), W_2_ffn(c_t)
...
第18层： W_18_attn(c_t), W_18_ffn(c_t)
```

所以：

\[
(\gamma_l,\beta_l,g_l)=W_lc_t
\]

输入时间条件相同，层专属的 `W_l` 不同，生成的调制量也不同。源码的 `nn.scan(variable_axes={"params": 0})` 正是在层维保存不同参数。

## 8. 高噪声和低噪声阶段

代码时间约定：

```text
t≈1：接近纯噪声
t≈0：接近目标动作
```

模型可能在高噪声阶段更依赖视觉语言条件、进行较大结构修正；在低噪声阶段更多保留当前动作并精修细节。这里的具体关系不是硬编码的，也不保证 gate 随时间单调变化，而是通过 Flow Matching Loss 学出来的。

## 9. 为什么 modulation Dense 零初始化

源码使用零初始化 kernel。训练开始时近似有：

```text
scale=0, shift=0, gate=0
```

于是：

```text
AdaRMS(x,t) = RMSNorm(x)
x_next = x + 0*F(x) = x
```

残差分支不会在初始化瞬间剧烈扰动特征；训练先学会逐渐打开门控，再学习各时间阶段的调制方式。这与常见的 adaptive normalization zero-init 思路一致。

## 10. 最终记忆公式

\[
\boxed{
x_{next}=x+g_l(t)\odot
F_l\left(
\operatorname{RMSNorm}(x)\odot(1+\gamma_l(t))+\beta_l(t)
\right)
}
\]

其中固定参数是 `F_l` 和产生调制量的投影层；动态变化的是由当前 `t` 计算出的 `γ/β/g` 和中间激活。

## 11. 阅读验收

1. 残差连接为何比 `x_next=F(x)` 更容易训练深网络？
2. gate 控制的是模型权重，还是残差分支的输出？
3. 同一个 `[B,1024]` 时间向量如何作用于 `[B,16,1024]`？
4. 一个 Block 为什么生成两组不同的 `scale/shift/gate`？
5. 18 层收到同一个 `c_t`，为什么调制结果不同？

下一章将训练公式和推理积分闭合起来。

