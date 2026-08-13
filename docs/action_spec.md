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

The physical action is clipped before converting to Isaac raw action. The
initial conservative component limits are 0.025 m translation and 0.12 rad
rotation per step. The adapter divides the first six dimensions by the Isaac
IK scale because Isaac's action term applies that scale internally.

Training conversion, replay, policy serving, residual composition and online
deployment must import the same adapter implementation.

