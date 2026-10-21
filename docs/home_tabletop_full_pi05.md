# 家庭桌面完整任务：π0.5 主导控制的录制规程

状态：代码与离线契约已准备；**尚未取得新的四阶段仿真录像**。不得用原版厨房长任务视频或脚本教师视频冒充本场景的 π0.5 成果。

家庭场景使用抬高到木桌面的 Franka／抽屉／药瓶工作空间；地板由仓库内 `assets/home_oak_floor.usda` 与 `assets/home_oak_floor.png` 提供真实木纹，而不是 Isaac 默认黑白网格。旧的 Isaac 4.5 Simple_Room 与 YCB 陈列资产不会在家庭场景加载。木纹是视觉材质，地面物理仍由默认地面承担，不改变策略的动作空间。

## 控制关系

同一条 Isaac episode 内执行 `OPEN → PICK → PLACE → CLOSE`，阶段由任务状态机选择，分别发送四条原有训练提示词。每 4 个控制步用桌面相机、腕部相机和机器人本体状态请求一次 π0.5 动作块。模型动作是主动作；DLS 参考动作仅向模型动作增加范数不超过 `0.10` 的 6D 修正，夹爪开合始终由模型决定。模型若不能完成，成功门槛会失败，不会静默切换到教师动作。

请注意：阶段判断和有界修正仍会用仿真真值，所以成功结果应称为 **π0.5-primary、truth-guided DLS-assisted**，不是纯视觉自主策略。真实视觉输入是 π0.5 的组成部分，不是一个独立的“视觉纠错器”。此处的运动学求解器名称是 **DLS**，不是 DLA。

## 运行前条件

- Isaac Sim 6.0.1、Isaac Lab 3.0.0-beta2 和仓库的 Isaac Python 环境。
- 可实际加载的四技能 π0.5 LoRA checkpoint；仓库发布的视频不是权重文件。
- 推荐两张 24 GB GPU：GPU 0 跑 Isaac/RTX 相机，GPU 1 跑 OpenPI/JAX 服务。若只有一张仿真 GPU，应将策略服务部署在另一台可连接的机器上，并设置 `POLICY_HOST`。
- `ffmpeg` 可用。输出目录留足 HDF5 三机位轨迹和 MP4 的空间。

双卡同机运行：

```bash
PI05_CHECKPOINT=/absolute/path/to/four_skill_lora_checkpoint \
  bash scripts/run_home_tabletop_full_pi05.sh
```

独立的模型服务器已经启动时：

```bash
POLICY_HOST=MODEL_SERVER_IP POLICY_PORT=8000 \
  bash scripts/run_home_tabletop_full_pi05.sh
```

可用 `DATASET`、`VIDEO` 指定输出，`DLS_MAX_CORRECTION_NORM` 指定修正上限。运行器先检查服务，然后录制，最后通过 `scripts/validate_home_full_pi05.py` 核验才合成视频。每次运行创建带时间戳的新文件，不覆盖既有轨迹。

## 发布门槛

验证器要求：恰好一条成功且未剪接的 episode；家庭桌面场景；`control_mode=policy_first`；三路相机、阶段、提示和动作逐帧对齐；`OPEN/PICK/PLACE/CLOSE` 顺序齐全；至少四次真实模型请求；所有 DLS 修正小于配置上限；最终 6D 动作等于模型动作加有界修正；夹爪动作等于模型动作；模型运动量占模型加修正总量的比例至少 60%。只有通过后才能将视频链接加入正式 README。

验证器衡量的是动作来源，不单独证明训练泛化；还须人工检查柜门/抽屉开闭、药瓶真正落入抽屉、三路相机有内容，以及末尾没有穿模或悬空。
