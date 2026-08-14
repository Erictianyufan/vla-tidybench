SHELL := /bin/bash

.PHONY: doctor test sim-smoke sim-camera-smoke drawer-scene-smoke drawer-open-smoke drawer-pick-smoke drawer-place-smoke drawer-close-smoke drawer-full-smoke drawer-replay drawer-atomic-validate drawer-convert-openpi drawer-norm-stats drawer-train drawer-policy-smoke drawer-policy-serve drawer-policy-run drawer-demo pick-rl-train pick-rl-record extension-smoke ood-plan protocol-smoke record scripted-smoke scripted-collect replay annotate mimic-smoke convert-openpi-smoke openpi-norm-stats openpi-data-smoke train-pi05-smoke package-demo prepublish

doctor:
	./scripts/remote_doctor.sh

test:
	PYTHONPATH=source /home/ubuntu/env_isaaclab/bin/python -m pytest tests

sim-smoke:
	./scripts/run_isaac.sh scripts/isaac_smoke.py \
		--task Isaac-Stack-Cube-Franka-IK-Rel-v0 --num_envs 1 --num_steps 40 --device cuda:0 --viz none

sim-camera-smoke:
	./scripts/run_isaac.sh scripts/isaac_smoke.py \
		--task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --num_envs 1 --num_steps 10 \
		--device cuda:0 --enable_cameras --viz none

drawer-scene-smoke:
	./scripts/run_isaac.sh scripts/smoke_drawer_task.py \
		--device cuda:0 --enable_cameras --viz none

drawer-open-smoke:
	SKILL=open NUM_DEMOS=1 MAX_ATTEMPTS=3 \
	DATASET_FILE=/home/ubuntu/data/vla-tidybench/raw/drawer_open_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-pick-smoke:
	SKILL=pick NUM_DEMOS=1 MAX_ATTEMPTS=3 \
	DATASET_FILE=/home/ubuntu/data/vla-tidybench/raw/drawer_pick_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-place-smoke:
	SKILL=place NUM_DEMOS=1 MAX_ATTEMPTS=5 MAX_STEPS=520 \
	DATASET_FILE=/home/ubuntu/data/vla-tidybench/raw/drawer_place_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-close-smoke:
	SKILL=close NUM_DEMOS=1 MAX_ATTEMPTS=3 \
	DATASET_FILE=/home/ubuntu/data/vla-tidybench/raw/drawer_close_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-full-smoke:
	SKILL=full NUM_DEMOS=1 MAX_ATTEMPTS=5 MAX_STEPS=900 \
	DATASET_FILE=/home/ubuntu/data/vla-tidybench/raw/drawer_full_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-replay:
	@test -n "$(DATASET_FILE)" -a -n "$(SKILL)" || \
		(echo "usage: make drawer-replay DATASET_FILE=/abs/demo.hdf5 SKILL=open" >&2; exit 2)
	DATASET_FILE="$(DATASET_FILE)" SKILL="$(SKILL)" ./scripts/replay_drawer_demos.sh

drawer-atomic-validate:
	./scripts/validate_drawer_atomic.sh

drawer-convert-openpi:
	./scripts/run_openpi.sh scripts/convert_stack_to_lerobot.py \
		--config configs/data/drawer_m2_smoke.json \
		--data-root /home/ubuntu/data/vla-tidybench/raw --overwrite

drawer-norm-stats:
	./scripts/run_openpi.sh scripts/compute_drawer_norm_stats.py

drawer-train:
	OPENPI_CUDA_VISIBLE_DEVICES=$${OPENPI_CUDA_VISIBLE_DEVICES:-0,1} \
	./scripts/run_openpi.sh scripts/train_drawer_pi05.py \
		--steps $${STEPS:-2} --batch-size $${BATCH_SIZE:-2} \
		--fsdp-devices $${FSDP_DEVICES:-2} \
		--exp-name $${EXP_NAME:-smoke} --overwrite

drawer-policy-smoke:
	OPENPI_CUDA_VISIBLE_DEVICES=$${POLICY_GPU:-1} ./scripts/run_openpi.sh scripts/smoke_drawer_policy.py \
		--checkpoint /home/ubuntu/data/vla-tidybench/checkpoints/openpi-runs/pi05_tidybench_drawer_lora/drawer-smoke/1 \
		--dataset /home/ubuntu/data/vla-tidybench/raw/drawer_open_smoke.hdf5

drawer-policy-serve:
	@test -n "$(CHECKPOINT)" || (echo "usage: make drawer-policy-serve CHECKPOINT=/abs/checkpoint" >&2; exit 2)
	OPENPI_CUDA_VISIBLE_DEVICES=$${POLICY_GPU:-1} ./scripts/run_openpi.sh scripts/serve_drawer_policy.py \
		--checkpoint "$(CHECKPOINT)" --port $${POLICY_PORT:-8000}

drawer-policy-run:
	./scripts/run_isaac.sh scripts/run_drawer_pi05_closed_loop.py \
		--device cuda:0 --enable_cameras --viz $${VIZ:-none} --port $${POLICY_PORT:-8000} \
		--output $${OUTPUT:-/home/ubuntu/data/vla-tidybench/eval/pi05_open_closed_loop.hdf5}

drawer-demo:
	FFMPEG_BIN=$$(./scripts/run_openpi.sh -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') \
	./scripts/run_openpi.sh scripts/render_skill_suite.py \
		--data-root /home/ubuntu/data/vla-tidybench/raw \
		--output artifacts/demo/vla-tidybench-skill-suite.mp4

pick-rl-train:
	./scripts/run_isaac.sh scripts/train_pick_residual_sac.py \
		--mode train --device cuda:0 --viz none --timesteps $${RL_STEPS:-200} \
		--checkpoint /home/ubuntu/data/vla-tidybench/checkpoints/pick_residual_sac_tomato \
		--metrics results/metrics/pick_residual_sac_tomato.json

pick-rl-record:
	./scripts/run_isaac.sh scripts/train_pick_residual_sac.py \
		--mode record --device cuda:0 --enable_cameras --viz none --showcase \
		--checkpoint /home/ubuntu/data/vla-tidybench/checkpoints/pick_residual_sac_tomato \
		--output /home/ubuntu/data/vla-tidybench/eval/pick_residual_tomato_rl_success.hdf5

extension-smoke:
	/home/ubuntu/env_isaaclab/bin/python scripts/validate_extension_configs.py

ood-plan:
	/home/ubuntu/env_isaaclab/bin/python scripts/plan_ood_eval.py

protocol-smoke:
	@set -euo pipefail; \
		./scripts/run_openpi.sh policy_bridge/dummy_policy_server.py --port 8001 > /tmp/vla-tidybench-policy.log 2>&1 & \
		server_pid=$$!; \
		trap 'kill $$server_pid 2>/dev/null || true' EXIT; \
		sleep 2; \
		./scripts/run_openpi.sh policy_bridge/protocol_smoke.py --port 8001

record:
	./scripts/record_stack_demos.sh

scripted-smoke:
	NUM_DEMOS=1 MAX_ATTEMPTS=4 SEED=41 \
	DATASET=/home/ubuntu/data/vla-tidybench/raw/stack_scripted_smoke.hdf5 \
	./scripts/collect_scripted_stack.sh --overwrite

scripted-collect:
	NUM_DEMOS=$${NUM_DEMOS:-7} MAX_ATTEMPTS=$${MAX_ATTEMPTS:-28} \
	DATASET=/home/ubuntu/data/vla-tidybench/raw/stack_scripted.hdf5 \
	./scripts/collect_scripted_stack.sh --overwrite

replay:
	./scripts/replay_stack_demos.sh

annotate:
	./scripts/annotate_stack_demos.sh

mimic-smoke:
	./scripts/generate_stack_mimic_smoke.sh

convert-openpi-smoke:
	./scripts/run_openpi.sh scripts/convert_stack_to_lerobot.py \
		--config configs/data/stack_m1_smoke.json \
		--data-root /home/ubuntu/data/vla-tidybench/raw \
		--overwrite

openpi-norm-stats:
	./scripts/run_openpi.sh scripts/compute_stack_norm_stats.py

openpi-data-smoke:
	./scripts/run_openpi.sh scripts/smoke_openpi_batch.py --batch-size 1

train-pi05-smoke:
	OPENPI_CUDA_VISIBLE_DEVICES=$${OPENPI_CUDA_VISIBLE_DEVICES:-0,1} \
	./scripts/run_openpi.sh scripts/train_stack_pi05_smoke.py \
		--steps $${STEPS:-2} --batch-size $${BATCH_SIZE:-2} \
		--fsdp-devices $${FSDP_DEVICES:-2} \
		--exp-name $${EXP_NAME:-smoke} --overwrite

package-demo:
	@test -n "$(INPUT)" || (echo "usage: make package-demo INPUT=/absolute/path/to/raw.mp4" >&2; exit 2)
	./scripts/package_demo.sh "$(INPUT)"

prepublish:
	/home/ubuntu/env_isaaclab/bin/python scripts/prepublish_check.py --require-clean
