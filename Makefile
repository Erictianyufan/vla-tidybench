SHELL := /bin/bash

.PHONY: doctor test sim-smoke sim-camera-smoke protocol-smoke record replay annotate mimic-smoke

doctor:
	./scripts/remote_doctor.sh

test:
	/home/ubuntu/env_isaaclab/bin/python -m pytest tests

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

replay:
	./scripts/replay_stack_demos.sh

annotate:
	./scripts/annotate_stack_demos.sh

mimic-smoke:
	./scripts/generate_stack_mimic_smoke.sh
