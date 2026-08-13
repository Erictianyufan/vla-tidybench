# Canonical Franka action specification

The only project action representation is:

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

- Translation: metres per policy step, robot base frame.
- Rotation: axis-angle rotation vector in radians per policy step, robot base frame.
- Gripper: positive is open, negative is close.
- Policy rate: 20 Hz (`sim.dt=0.01`, `decimation=5`).
- Quaternion order in Isaac Lab 3 data: XYZW.
- End-effector body: `panda_hand`, tool offset `[0, 0, 0.107]` m.
- Isaac differential IK mode: relative pose, DLS, internal scale `0.5`.

The physical action is clipped before converting to Isaac raw action. For this
simulation task the replay-calibrated component limits are 0.15 m translation
and 0.55 rad rotation per policy step, followed by vector-norm limits of 0.18 m
and 0.75 rad. These values bound the replay-validated expert envelope; they are
not real-robot safety limits. A diagnostic replay with the earlier placeholder
limits (0.025 m and 0.12 rad) failed 0/10 because it altered 28.8% of the
candidate actions. The adapter divides the first six dimensions by the Isaac IK
scale because Isaac's action term applies that scale internally.

Training conversion, replay, policy serving, residual composition and online
deployment must import the same adapter implementation.
