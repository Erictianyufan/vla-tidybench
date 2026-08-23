# ML, Robotics, and VLA Source Reading

Read this reference only when the repository contains learned models, papers, tensor pipelines, robotics semantics, diffusion/flow models, or VLA systems.

## Paper → framework → code

Avoid explaining the paper and source as separate stories. For each claimed module, build a mapping:

| Paper/framework concept | Public-code module | Input/output | Implemented here? |
|---|---|---|---:|

Distinguish:

- paper architecture from the checked-out revision;
- public inference/fine-tuning code from private pretraining infrastructure;
- conceptual names from actual class/function names;
- equations' time/sign conventions from code conventions.

If a paper statement is not represented in public code, say so directly.

## Start with data semantics

Before model layers, identify:

```text
raw sensor/record fields
physical state and action definitions
repack/mapping transforms
normalization statistics
augmentation
tokenization/discretization
padding/model width
batch and sequence construction
output unnormalization/adaptation
```

For robotics, maintain separate notation for:

- physical state dimension;
- physical action dimension;
- model padded action width;
- action horizon/control rate;
- coordinate frame and absolute/delta semantics;
- privileged simulator truth versus deployable observation.

Never infer physical controllability from a padded model dimension.

## Tensor tracking

Define notation once, for example:

```text
B batch
T token/sequence length
H action horizon
D model width
Da physical action dimension
```

At every important operation state both:

1. shape change;
2. semantic change.

Example:

```text
[B,H,Da] --pad--> [B,H,D]
```

changes shape but not physical degrees of freedom, while normalization preserves shape but changes numeric semantics.

## Attention and masks

Teach attention in this order:

1. what each token represents;
2. Query/Key/Value roles;
3. sequence/block layout;
4. mask construction;
5. allowed information-flow matrix;
6. cache behavior;
7. multi-head/MQA/GQA dimensions if relevant.

For boundary masks, compute a tiny concrete sequence and its block IDs instead of relying on names such as `ar_mask`. Explicitly state whether generation is token-autoregressive, block-causal, bidirectional within a chunk, or something else.

## Residuals and adaptive normalization

Before code, establish:

```text
ordinary layer: x_next = F(x)
residual layer: x_next = x + F(x)
gated residual: x_next = x + g(c) * F(x)
```

For adaptive normalization, distinguish fixed learned weights from dynamically generated scale/shift/gate. Track condition and activation shapes, broadcasting, per-layer parameter independence, and initialization behavior.

## Generative training/inference closure

For diffusion, flow matching, autoregressive, or energy-based models, keep four items separate:

```text
clean target
corrupted/intermediate input
supervision target
model prediction
```

Derive the target from the implemented interpolation/noise schedule. Then show how inference reverses or integrates that process. Verify sign and time direction with a one-dimensional numeric example.

Always state whether the model predicts:

- final sample/action;
- noise;
- score;
- velocity/vector field;
- discrete tokens;
- distribution parameters.

## Training runtime

Trace:

```text
per-element loss
  -> scalar reduction
  -> autodiff
  -> trainable/frozen filter
  -> gradient clipping/optimizer
  -> parameter update
  -> EMA/checkpoint
```

Do not confuse model velocity symbols with optimizer moment variables. Verify whether EMA, LoRA, mixed precision, FSDP, gradient checkpointing, or cache reuse is enabled in the concrete project rather than merely supported by the framework.

## Recommended lesson order for VLA repositories

1. policy/inference wrapper;
2. dataset and transforms;
3. observation/action contracts;
4. high-level model architecture;
5. prefix/condition encoder;
6. action/generation path;
7. attention and masks;
8. conditioning, residual, and normalization mechanisms;
9. training objective;
10. sampling/integration and train–infer closure;
11. optimizer/training runtime;
12. serving, action execution, evaluation, and safety boundaries.

Keep alternative model families and third-party benchmarks out of the first pass unless they are on the user's primary path.
