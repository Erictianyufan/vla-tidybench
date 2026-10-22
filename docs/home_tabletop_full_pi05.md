# 家庭桌面完整任务：π0.5 主导控制的录制规程

状态：已取得一条通过动作来源审计和物理成功判据的[π0.5 在线闭环录像](media/home-tabletop-full-pi05-live.mp4)；[GitHub Release](https://github.com/Erictianyufan/vla-tidybench/releases/tag/v0.2.0-home-pi05)提供原始 HDF5、MP4、GIF 和指标 JSON 下载。四阶段 `OPEN → PICK → PLACE → CLOSE` 在同一个 425 步 episode 内完成；3 次尝试中第 3 次成功。早期[脚本教师预览](media/home-tabletop-full-teacher-wood-preview.mp4)继续保留，但不得与本次模型结果混称。

家庭场景使用抬高到木桌面的 Franka／抽屉／药瓶工作空间；地板由仓库内 `assets/home_oak_floor.usda` 与 `assets/home_oak_floor.png` 提供真实木纹，而不是 Isaac 默认黑白网格。旧的 Isaac 4.5 Simple_Room 与 YCB 陈列资产不会在家庭场景加载。木纹是视觉材质，地面物理仍由默认地面承担，不改变策略的动作空间。

## 控制关系

同一条 Isaac episode 内执行 `OPEN → PICK → PLACE → CLOSE`，阶段由任务状态机选择，分别发送四条训练提示词。每 4 个控制步用桌面相机、腕部相机和 **18D 相对关节位置/速度** 请求一次 π0.5 动作块；不能把 Isaac 的绝对关节角直接送给使用 `joint_pos_rel` / `joint_vel_rel` 训练的模型。模型动作是主动作；DLS 参考动作仅向模型动作增加范数不超过 `0.20` 的 6D 修正。夹爪开合完全由模型决定，连续闭合 6 步或张开 4 步才切换执行状态；模型若不能完成，成功门槛会失败，不会静默切换到教师动作。

请注意：阶段判断和有界修正仍会用仿真真值，所以成功结果应称为 **π0.5-primary、truth-guided DLS-assisted**，不是纯视觉自主策略。真实视觉输入是 π0.5 的组成部分，不是一个独立的“视觉纠错器”。此处的运动学求解器名称是 **DLS**，不是 DLA。

## 运行前条件

- Isaac Sim 6.0.1、Isaac Lab 3.0.0-beta2 和仓库的 Isaac Python 环境。
- 可实际加载的四技能 π0.5 checkpoint。本次成功使用 action-expert 微调模型，1000 步 checkpoint 指纹 `2308fdd23d3dfc9c0ba52019e874d4b2d00c836a1e325c3e459ed83bd8930b0d`；[训练完成记录](../results/metrics/home_relative_release_training_completion.json)与[数据来源记录](../results/metrics/home_relative_release_dataset_provenance.json)可核对。此前公开的[第三阶段完整权重](https://github.com/Erictianyufan/vla-tidybench-pi05-weights)是上游实验，**不是**这条视频的权重。
- 本次部署实测在单张 48 GB RTX 4090 上同时运行策略服务与 Isaac 相机仿真；若改用 12 GB 显卡，应分机部署，不能把本次资源结果直接外推。
- `ffmpeg` 可用。输出目录留足 HDF5 三机位轨迹和 MP4 的空间。

双卡同机运行：

```bash
PI05_CHECKPOINT=/absolute/path/to/four_skill_lora_checkpoint \
  bash scripts/run_home_tabletop_full_pi05.sh
```

上面的同机脚本使用当前分支的 LoRA 服务端，**不能直接加载本次 action-expert 权重**。本次成功的策略服务来自训练时的项目提交 `50f868349961e04357356f54e47069642f95abb6`、OpenPI 提交 `15a9616a00943ada6c20a0f158e3adb39df2ccac`，以 `--mode expert --four-skill --dataset-repo scuee_user06/vla_tidybench_home_relative_release_vm20261023` 加载成功 checkpoint `.../home_relative_release_expert_1001_vm20261023j/1000`。训练数据集指纹 `671389aa572354c61ff0e6b8e7817ffe77b2b661b910ee652d9516d161bc7b85`；其中失败轨迹带教师动作标签，仅用于训练，不作为成功样本。

独立的模型服务器已经启动时：

```bash
POLICY_HOST=MODEL_SERVER_IP POLICY_PORT=8000 \
  bash scripts/run_home_tabletop_full_pi05.sh
```

可用 `DATASET`、`VIDEO` 指定输出，`DLS_MAX_CORRECTION_NORM=0.20`、`DLS_CORRECTION_MODE=skill_router` 和 `POLICY_GRIPPER_OPEN_CONFIRM_STEPS=4` 复现本次动作门槛。运行器先检查服务，然后录制，最后通过 `scripts/validate_home_full_pi05.py` 核验才合成视频。每次运行创建带时间戳的新文件，不覆盖既有轨迹。采样存在随机性；固定 `seed=610` 不保证每次都在相同尝试序号成功。

## 发布门槛

验证器要求：恰好一条成功且未剪接的 episode；家庭桌面场景；`control_mode=policy_first`；相对关节状态约定；三路相机、阶段、提示和动作逐帧对齐；`OPEN/PICK/PLACE/CLOSE` 顺序齐全；至少四次真实模型请求；所有 DLS 修正小于配置上限；最终 6D 动作等于模型动作加有界修正；夹爪动作等于模型动作；模型运动量占模型加修正总量的比例至少 60%。本次[审计结果](../results/metrics/home_tabletop_full_pi05_live.json)：425 帧、108 次推理、模型运动量占比 `0.7860`、最大 DLS 修正 `0.2000`、平均推理 `126.34 ms`，全部通过。

验证器衡量的是动作来源，不单独证明训练泛化；还须人工检查柜门/抽屉开闭、药瓶真正落入抽屉、三路相机有内容，以及末尾没有穿模或悬空。
