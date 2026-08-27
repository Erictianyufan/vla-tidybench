SHELL := /bin/bash
VLA_TIDYBENCH_DATA ?= /data/$(USER)/vla-tidybench
FOUR_SKILL_FLAG ?=
MODE ?= lora
TRAIN_STATE_FLAG ?= --overwrite
POLICY_MODE ?= full
POLICY_CONFIG_FLAG ?= --four-skill
PI05_MAIN_CONFIG ?= configs/data/drawer_four_skill_formal.json
PI05_HARD_CONFIG ?= configs/data/drawer_four_skill_hard_recovery.json
PI05_REPO_PREFIX ?= $(USER)/vla_tidybench_drawer_v1

.PHONY: doctor test sim-smoke sim-camera-smoke drawer-scene-smoke drawer-open-smoke drawer-pick-smoke drawer-place-smoke drawer-close-smoke drawer-full-smoke drawer-replay drawer-atomic-validate drawer-convert-openpi drawer-norm-stats drawer-four-skill-norm-stats drawer-train drawer-train-lora drawer-train-full drawer-four-skill-train-lora drawer-four-skill-train-full pi05-plan-data pi05-convert-data pi05-prepare-norm-stats pi05-formal-prepare pi05-formal-pipeline pi05-three-stage-synthetic-smoke pi05-three-stage-smoke pi05-three-stage-train pi05-eval-suite pi05-export-final pi05-verify-deployment pi05-deployment-serve pi05-policy-probe drawer-policy-smoke drawer-policy-serve drawer-policy-run drawer-demo continuous-medicine-demo pick-rl-train pick-rl-record media-gifs extension-smoke ood-plan protocol-smoke record scripted-smoke scripted-collect replay annotate mimic-smoke convert-openpi-smoke openpi-norm-stats openpi-data-smoke train-pi05-smoke package-demo prepublish

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
	DATASET_FILE=$(VLA_TIDYBENCH_DATA)/raw/drawer_open_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-pick-smoke:
	SKILL=pick NUM_DEMOS=1 MAX_ATTEMPTS=3 \
	DATASET_FILE=$(VLA_TIDYBENCH_DATA)/raw/drawer_pick_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-place-smoke:
	SKILL=place NUM_DEMOS=1 MAX_ATTEMPTS=5 MAX_STEPS=520 \
	DATASET_FILE=$(VLA_TIDYBENCH_DATA)/raw/drawer_place_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-close-smoke:
	SKILL=close NUM_DEMOS=1 MAX_ATTEMPTS=3 \
	DATASET_FILE=$(VLA_TIDYBENCH_DATA)/raw/drawer_close_smoke.hdf5 \
	./scripts/collect_scripted_drawer.sh --overwrite

drawer-full-smoke:
	SKILL=full NUM_DEMOS=1 MAX_ATTEMPTS=5 MAX_STEPS=900 \
	DATASET_FILE=$(VLA_TIDYBENCH_DATA)/raw/drawer_full_smoke.hdf5 \
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
		--data-root $(VLA_TIDYBENCH_DATA)/raw --overwrite

drawer-norm-stats:
	./scripts/run_openpi.sh scripts/compute_drawer_norm_stats.py

drawer-four-skill-norm-stats:
	./scripts/run_openpi.sh scripts/compute_drawer_norm_stats.py --four-skill

pi05-plan-data:
	@test -n "$(MAIN_SOURCE_CONFIG)" -a -n "$(HARD_SOURCE_CONFIG)" -a -n "$(REPO_PREFIX)" || \
		(echo "usage: make pi05-plan-data MAIN_SOURCE_CONFIG=configs/data/main.json HARD_SOURCE_CONFIG=configs/data/hard.json REPO_PREFIX=owner/name" >&2; exit 2)
	./scripts/run_openpi.sh scripts/plan_pi05_data_splits.py \
		--main-config "$(MAIN_SOURCE_CONFIG)" --hard-config "$(HARD_SOURCE_CONFIG)" \
		--data-root $(VLA_TIDYBENCH_DATA)/raw \
		--output-dir $${MANIFEST_DIR:-$(VLA_TIDYBENCH_DATA)/manifests/pi05-formal} \
		--repo-prefix "$(REPO_PREFIX)" --overwrite

pi05-convert-data:
	@manifest_dir=$${MANIFEST_DIR:-$(VLA_TIDYBENCH_DATA)/manifests/pi05-formal}; \
	for manifest in main_train.json main_validation.json hard_validation.json hard_mix_train.json; do \
		test -f "$$manifest_dir/$$manifest" || { echo "missing $$manifest_dir/$$manifest" >&2; exit 2; }; \
		./scripts/run_openpi.sh scripts/convert_stack_to_lerobot.py \
			--config "$$manifest_dir/$$manifest" --data-root $(VLA_TIDYBENCH_DATA)/raw --overwrite || exit $$?; \
	done

pi05-prepare-norm-stats:
	@test -n "$(MAIN_DATASET_REPO)" -a -n "$(HARD_DATASET_REPO)" || \
		(echo "usage: make pi05-prepare-norm-stats MAIN_DATASET_REPO=org/data HARD_DATASET_REPO=org/hard-mix" >&2; exit 2)
	VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID="$(MAIN_DATASET_REPO)" \
		./scripts/run_openpi.sh scripts/compute_drawer_norm_stats.py --four-skill --mode lora
	VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID="$(MAIN_DATASET_REPO)" \
		./scripts/run_openpi.sh scripts/compute_drawer_norm_stats.py --four-skill --mode full
	VLA_TIDYBENCH_DRAWER_FOUR_SKILL_REPO_ID="$(HARD_DATASET_REPO)" \
		./scripts/run_openpi.sh scripts/compute_drawer_norm_stats.py --four-skill --mode full

pi05-formal-prepare:
	$(MAKE) pi05-plan-data \
		MAIN_SOURCE_CONFIG="$(PI05_MAIN_CONFIG)" HARD_SOURCE_CONFIG="$(PI05_HARD_CONFIG)" \
		REPO_PREFIX="$(PI05_REPO_PREFIX)"
	$(MAKE) pi05-convert-data
	$(MAKE) pi05-prepare-norm-stats \
		MAIN_DATASET_REPO="$(PI05_REPO_PREFIX)_train" \
		HARD_DATASET_REPO="$(PI05_REPO_PREFIX)_hard_mix"

pi05-formal-pipeline: pi05-formal-prepare
	$(MAKE) pi05-three-stage-train \
		MAIN_DATASET_REPO="$(PI05_REPO_PREFIX)_train" \
		HARD_DATASET_REPO="$(PI05_REPO_PREFIX)_hard_mix" \
		TRAIN_STATE_FLAG="$(TRAIN_STATE_FLAG)"

drawer-train:
	OPENPI_CUDA_VISIBLE_DEVICES=$${OPENPI_CUDA_VISIBLE_DEVICES:-0,1,2} \
	./scripts/run_openpi.sh scripts/train_drawer_pi05.py \
		--mode $(MODE) --steps $${STEPS:-2} --batch-size $${BATCH_SIZE:-3} \
		--fsdp-devices $${FSDP_DEVICES:-3} \
		--exp-name $${EXP_NAME:-smoke} $(FOUR_SKILL_FLAG) $(TRAIN_STATE_FLAG)

drawer-train-lora:
	$(MAKE) drawer-train MODE=lora

drawer-train-full:
	PI05_FSDP_MIN_SIZE_MBYTES=0 $(MAKE) drawer-train MODE=full

drawer-four-skill-train-lora:
	$(MAKE) drawer-train MODE=lora FOUR_SKILL_FLAG=--four-skill

drawer-four-skill-train-full:
	PI05_FSDP_MIN_SIZE_MBYTES=0 $(MAKE) drawer-train MODE=full FOUR_SKILL_FLAG=--four-skill

pi05-three-stage-synthetic-smoke:
	OPENPI_CUDA_VISIBLE_DEVICES=$${OPENPI_CUDA_VISIBLE_DEVICES:-0,1,2} \
	XLA_PYTHON_CLIENT_ALLOCATOR=$${XLA_PYTHON_CLIENT_ALLOCATOR:-platform} \
	./scripts/run_openpi.sh scripts/run_pi05_three_stage.py \
		--smoke --synthetic-data --stage all --overwrite

pi05-three-stage-smoke:
	OPENPI_CUDA_VISIBLE_DEVICES=$${OPENPI_CUDA_VISIBLE_DEVICES:-0,1,2} \
	XLA_PYTHON_CLIENT_ALLOCATOR=$${XLA_PYTHON_CLIENT_ALLOCATOR:-platform} \
	./scripts/run_openpi.sh scripts/run_pi05_three_stage.py --smoke --stage all --overwrite

pi05-three-stage-train:
	@test -n "$(MAIN_DATASET_REPO)" -a -n "$(HARD_DATASET_REPO)" || \
		(echo "usage: make pi05-three-stage-train MAIN_DATASET_REPO=org/data HARD_DATASET_REPO=org/hard-mix" >&2; exit 2)
	OPENPI_CUDA_VISIBLE_DEVICES=$${OPENPI_CUDA_VISIBLE_DEVICES:-0,1,2} \
	XLA_PYTHON_CLIENT_ALLOCATOR=$${XLA_PYTHON_CLIENT_ALLOCATOR:-platform} \
	./scripts/run_openpi.sh scripts/run_pi05_three_stage.py --stage all \
		--main-dataset-repo "$(MAIN_DATASET_REPO)" --hard-dataset-repo "$(HARD_DATASET_REPO)" \
		--batch-size $${BATCH_SIZE:-3} --fsdp-devices $${FSDP_DEVICES:-3} $(TRAIN_STATE_FLAG)

pi05-export-final:
	@test -n "$(CHECKPOINT)" -a -n "$(DATASET_REPO)" -a -n "$(EVAL_REPORT)" || \
		(echo "usage: make pi05-export-final CHECKPOINT=/abs/step DATASET_REPO=org/hard-mix EVAL_REPORT=/abs/evaluation.json" >&2; exit 2)
	./scripts/run_openpi.sh scripts/export_pi05_checkpoint.py \
		--checkpoint "$(CHECKPOINT)" --dataset-repo "$(DATASET_REPO)" \
		--evaluation-report "$(EVAL_REPORT)" --mode $${EXPORT_POLICY_MODE:-full} --replace

pi05-verify-deployment:
	@test -n "$(DEPLOYMENT)" || \
		(echo "usage: make pi05-verify-deployment DEPLOYMENT=/abs/pi05-tidybench-final" >&2; exit 2)
	./scripts/run_openpi.sh scripts/verify_pi05_deployment.py \
		--deployment "$(DEPLOYMENT)" $${DEPLOYMENT_VALIDATION_FLAG:-}

pi05-deployment-serve: pi05-verify-deployment
	OPENPI_CUDA_VISIBLE_DEVICES=$${POLICY_GPU:-1} ./scripts/run_openpi.sh scripts/serve_drawer_policy.py \
		--deployment "$(DEPLOYMENT)" --port $${POLICY_PORT:-8000} \
		$${DEPLOYMENT_SERVE_FLAG:-}

pi05-policy-probe:
	./scripts/run_openpi.sh scripts/probe_drawer_policy.py \
		--host $${POLICY_HOST:-127.0.0.1} --port $${POLICY_PORT:-8000} \
		--runs $${PROBE_RUNS:-2} --expect-mode $${POLICY_MODE:-full} \
		$${POLICY_PROBE_FLAG:-}

pi05-eval-suite:
	./scripts/run_pi05_eval_suite.py \
		--output-root $${EVAL_ROOT:-$(VLA_TIDYBENCH_DATA)/eval/pi05-formal} \
		--context-manifest $${EVAL_CONTEXT_MANIFEST:-$(VLA_TIDYBENCH_DATA)/manifests/pi05-formal/main_validation.json} \
		--data-root $${EVAL_DATA_ROOT:-$(VLA_TIDYBENCH_DATA)/raw} \
		--host $${POLICY_HOST:-127.0.0.1} --port $${POLICY_PORT:-8000} \
		--min-success-rate $${MIN_SUCCESS_RATE:-0.6} \
		--max-p95-infer-ms $${MAX_P95_INFER_MS:-250} $${EVAL_STATE_FLAG:---overwrite}

drawer-policy-smoke:
	OPENPI_CUDA_VISIBLE_DEVICES=$${POLICY_GPU:-1} ./scripts/run_openpi.sh scripts/smoke_drawer_policy.py \
		--checkpoint $${CHECKPOINT:-$(VLA_TIDYBENCH_DATA)/checkpoints/openpi-runs/pi05_tidybench_drawer_lora/drawer-smoke/1} \
		$${POLICY_INPUT_FLAG:---dataset $(VLA_TIDYBENCH_DATA)/raw/drawer_open_smoke.hdf5} \
		--mode $(POLICY_MODE) $(POLICY_CONFIG_FLAG)

drawer-policy-serve:
	@test -n "$(CHECKPOINT)" || (echo "usage: make drawer-policy-serve CHECKPOINT=/abs/checkpoint" >&2; exit 2)
	OPENPI_CUDA_VISIBLE_DEVICES=$${POLICY_GPU:-1} ./scripts/run_openpi.sh scripts/serve_drawer_policy.py \
		--checkpoint "$(CHECKPOINT)" --mode $(POLICY_MODE) $(POLICY_CONFIG_FLAG) \
		--port $${POLICY_PORT:-8000}

drawer-policy-run:
	./scripts/run_isaac.sh scripts/run_drawer_pi05_closed_loop.py \
		--device cuda:0 --enable_cameras --viz $${VIZ:-none} \
		--host $${POLICY_HOST:-127.0.0.1} --port $${POLICY_PORT:-8000} \
		--skill $${POLICY_SKILL:-open} \
		--output $${OUTPUT:-$(VLA_TIDYBENCH_DATA)/eval/pi05_open_closed_loop.hdf5}

drawer-demo:
	FFMPEG_BIN=$$(./scripts/run_openpi.sh -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') \
	./scripts/run_openpi.sh scripts/render_skill_suite.py \
		--data-root $(VLA_TIDYBENCH_DATA)/raw \
		--output artifacts/demo/vla-tidybench-skill-suite.mp4

continuous-medicine-demo:
	./scripts/run_isaac.sh scripts/collect_scripted_drawer.py \
		--skill full --num_demos 1 --max_attempts 1 --max_steps 650 --seed 610 --overwrite \
		--dataset_file $(VLA_TIDYBENCH_DATA)/eval/drawer_full_medicine_pi05_live.hdf5 \
		--showcase --policy-host $${POLICY_HOST:-127.0.0.1} --policy-port $${POLICY_PORT:-8000} \
		--policy-residual-weight $${POLICY_RESIDUAL_WEIGHT:-0.0001} --policy-replan-steps 4 \
		--device cuda:0 --enable_cameras --viz none

pick-rl-train:
	./scripts/run_isaac.sh scripts/train_pick_residual_sac.py \
		--mode train --device cuda:0 --viz none --timesteps $${RL_STEPS:-200} \
		--checkpoint $(VLA_TIDYBENCH_DATA)/checkpoints/pick_residual_sac_tomato \
		--metrics results/metrics/pick_residual_sac_tomato.json

pick-rl-record:
	./scripts/run_isaac.sh scripts/train_pick_residual_sac.py \
		--mode record --device cuda:0 --enable_cameras --viz none --showcase \
		--checkpoint $(VLA_TIDYBENCH_DATA)/checkpoints/pick_residual_sac_tomato \
		--output $(VLA_TIDYBENCH_DATA)/eval/pick_residual_tomato_rl_success.hdf5

media-gifs:
	./scripts/run_openpi.sh scripts/render_all_video_gifs.py --media-dir docs/media

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
	DATASET=$(VLA_TIDYBENCH_DATA)/raw/stack_scripted_smoke.hdf5 \
	./scripts/collect_scripted_stack.sh --overwrite

scripted-collect:
	NUM_DEMOS=$${NUM_DEMOS:-7} MAX_ATTEMPTS=$${MAX_ATTEMPTS:-28} \
	DATASET=$(VLA_TIDYBENCH_DATA)/raw/stack_scripted.hdf5 \
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
		--data-root $(VLA_TIDYBENCH_DATA)/raw \
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
