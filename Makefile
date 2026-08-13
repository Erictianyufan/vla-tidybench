SHELL := /bin/bash

.PHONY: doctor test sim-smoke sim-camera-smoke protocol-smoke record scripted-smoke scripted-collect replay annotate mimic-smoke convert-openpi-smoke openpi-norm-stats openpi-data-smoke train-pi05-smoke package-demo prepublish

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
