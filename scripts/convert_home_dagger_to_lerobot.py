#!/usr/bin/env python3
"""Build a training-only LeRobot set from home demos and failed on-policy states.

Failed states retain their failure provenance; their action targets are truth
teacher corrections, not actions claimed to have succeeded on the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from vla_tidybench.data.isaac_hdf5 import canonical_actions, deployable_state, load_episode, sorted_episode_names


def add_episode(dataset: LeRobotDataset, *, images, wrist_images, states, actions, prompt: str) -> int:
    if not (len(images) == len(wrist_images) == len(states) == len(actions)):
        raise ValueError("image/state/action lengths differ")
    for index in range(len(actions)):
        dataset.add_frame(
            {
                "image": images[index],
                "wrist_image": wrist_images[index],
                "state": states[index],
                "actions": actions[index],
                "task": prompt,
            }
        )
    dataset.save_episode()
    return len(actions)


def add_focused_runs(
    dataset: LeRobotDataset, *, selected, images, wrist_images, states, actions, prompts, repeat: int
) -> tuple[int, int]:
    """Repeat contiguous wrong-gripper states without joining different prompts."""

    if not len(selected) or not repeat:
        return 0, 0
    starts = [0] + [
        i
        for i in range(1, len(selected))
        if selected[i] != selected[i - 1] + 1 or prompts[selected[i]] != prompts[selected[i - 1]]
    ]
    ends = starts[1:] + [len(selected)]
    episodes = frames = 0
    for _ in range(repeat):
        for first, last in zip(starts, ends, strict=True):
            indices = selected[first:last]
            frames += add_episode(
                dataset,
                images=images[indices],
                wrist_images=wrist_images[indices],
                states=states[indices],
                actions=actions[indices],
                prompt=prompts[int(indices[0])],
            )
            episodes += 1
    return episodes, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--dagger", type=Path, action="append", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dagger-repeat", type=int, default=3)
    parser.add_argument("--max-dagger-frames", type=int, default=150)
    parser.add_argument("--closed-state-repeat", type=int, default=0)
    parser.add_argument("--closed-state-max-frames", type=int, default=60)
    parser.add_argument("--premature-close-repeat", type=int, default=0)
    parser.add_argument("--premature-close-max-frames", type=int, default=60)
    parser.add_argument("--pick-approach-repeat", type=int, default=0)
    parser.add_argument("--pick-approach-max-frames", type=int, default=150)
    parser.add_argument("--handle-release-repeat", type=int, default=0)
    parser.add_argument("--handle-release-max-frames", type=int, default=100)
    parser.add_argument("--pick-grasp-repeat", type=int, default=0)
    parser.add_argument("--pick-grasp-max-frames", type=int, default=120)
    parser.add_argument("--place-release-repeat", type=int, default=0)
    parser.add_argument("--place-release-max-frames", type=int, default=100)
    args = parser.parse_args()
    if (
        min(
            args.dagger_repeat,
            args.max_dagger_frames,
            args.closed_state_max_frames,
            args.premature_close_max_frames,
            args.pick_approach_max_frames,
            args.handle_release_max_frames,
            args.pick_grasp_max_frames,
            args.place_release_max_frames,
        )
        < 1
        or min(
            args.closed_state_repeat,
            args.premature_close_repeat,
            args.pick_approach_repeat,
            args.handle_release_repeat,
            args.pick_grasp_repeat,
            args.place_release_repeat,
        )
        < 0
    ):
        parser.error("repeat and maximum frames must be positive; focused repeats may be zero")
    output = HF_LEROBOT_HOME / args.repo_id
    if output.exists():
        parser.error(f"dataset already exists; refusing to overwrite {output}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=20,
        features={
            "image": {"dtype": "image", "shape": (200, 200, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (200, 200, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (18,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=0,
    )
    nominal_episodes = nominal_frames = dagger_episodes = dagger_frames = 0
    closed_focus_episodes = closed_focus_frames = 0
    premature_close_episodes = premature_close_frames = 0
    pick_approach_episodes = pick_approach_frames = 0
    handle_release_episodes = handle_release_frames = 0
    pick_grasp_episodes = pick_grasp_frames = 0
    place_release_episodes = place_release_frames = 0
    with h5py.File(args.nominal, "r") as nominal:
        if not bool(nominal.attrs.get("derived_training_only", False)):
            raise ValueError("nominal input must be verified training-only home segments")
        for name in sorted_episode_names(nominal["data"]):
            episode = load_episode(args.nominal, name)
            prompt = str(nominal["data"][name].attrs["language_instruction"])
            nominal_frames += add_episode(
                dataset,
                images=episode.table_image,
                wrist_images=episode.wrist_image,
                states=episode.state,
                actions=episode.actions,
                prompt=prompt,
            )
            nominal_episodes += 1

    for dagger_path in args.dagger:
        with h5py.File(dagger_path, "r") as dagger:
            if not bool(dagger.attrs.get("training_only", False)):
                raise ValueError("DAgger input must be marked training-only")
            for name in sorted_episode_names(dagger["data"]):
                episode = dagger["data"][name]
                if (
                    bool(episode.attrs.get("success", True))
                    or episode.attrs.get("source") != "on_policy_truth_teacher_labeled_failure"
                ):
                    raise ValueError(f"expected a truth-labeled failure, got {dagger_path}::{name}")
                if episode.attrs.get("state_convention") != "joint_pos_rel_joint_vel_rel":
                    raise ValueError(f"DAgger state is not confirmed relative; refusing {dagger_path}::{name}")
                limit = min(args.max_dagger_frames, len(episode["actions"]))
                obs = episode["obs"]
                images = np.asarray(obs["table_cam"][:limit], dtype=np.uint8)
                wrist_images = np.asarray(obs["wrist_cam"][:limit], dtype=np.uint8)
                states = deployable_state(obs["joint_pos"][:limit], obs["joint_vel"][:limit])
                actions = canonical_actions(episode["actions"][:limit])
                prompts = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in episode["policy_prompts"][:limit]
                ]
                phases = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in episode["teacher_phase"][:limit]
                ]
                if len(prompts) != limit:
                    raise ValueError(f"prompt/action length mismatch in {dagger_path}::{name}")
                if len(phases) != limit:
                    raise ValueError(f"phase/action length mismatch in {dagger_path}::{name}")
                boundaries = (
                    [0] + [index for index in range(1, limit) if prompts[index] != prompts[index - 1]] + [limit]
                )
                for _ in range(args.dagger_repeat):
                    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                        dagger_frames += add_episode(
                            dataset,
                            images=images[start:end],
                            wrist_images=wrist_images[start:end],
                            states=states[start:end],
                            actions=actions[start:end],
                            prompt=prompts[start],
                        )
                        dagger_episodes += 1
                print(
                    f"added truth-labeled failure {dagger_path}::{name}: {limit} frames x{args.dagger_repeat}",
                    flush=True,
                )

                focus = np.flatnonzero(
                    np.asarray([phase in ("HANDLE_CLOSE", "DRAWER_PULL") for phase in phases]) & (actions[:, 6] < 0)
                )[: args.closed_state_max_frames]
                if len(focus) and args.closed_state_repeat:
                    starts = [0] + [
                        i
                        for i in range(1, len(focus))
                        if focus[i] != focus[i - 1] + 1 or prompts[focus[i]] != prompts[focus[i - 1]]
                    ]
                    ends = starts[1:] + [len(focus)]
                    for _ in range(args.closed_state_repeat):
                        for first, last in zip(starts, ends, strict=True):
                            selected = focus[first:last]
                            closed_focus_frames += add_episode(
                                dataset,
                                images=images[selected],
                                wrist_images=wrist_images[selected],
                                states=states[selected],
                                actions=actions[selected],
                                prompt=prompts[int(selected[0])],
                            )
                            closed_focus_episodes += 1
                    print(
                        f"added early closed-gripper corrective states {dagger_path}::{name}: "
                        f"{len(focus)} frames x{args.closed_state_repeat}",
                        flush=True,
                    )

                if args.premature_close_repeat:
                    if "executed_actions" not in episode:
                        raise ValueError(f"premature-close focus needs executed_actions in {dagger_path}::{name}")
                    executed = canonical_actions(episode["executed_actions"][:limit])
                    premature = np.flatnonzero(
                        np.asarray([phase in ("HANDLE_APPROACH", "HANDLE_ALIGN", "HANDLE_CLOSE") for phase in phases])
                        & (actions[:, 6] > 0)
                        & (executed[:, 6] < 0)
                    )[: args.premature_close_max_frames]
                    if len(premature):
                        starts = [0] + [
                            i
                            for i in range(1, len(premature))
                            if premature[i] != premature[i - 1] + 1
                            or prompts[premature[i]] != prompts[premature[i - 1]]
                        ]
                        ends = starts[1:] + [len(premature)]
                        for _ in range(args.premature_close_repeat):
                            for first, last in zip(starts, ends, strict=True):
                                selected = premature[first:last]
                                premature_close_frames += add_episode(
                                    dataset,
                                    images=images[selected],
                                    wrist_images=wrist_images[selected],
                                    states=states[selected],
                                    actions=actions[selected],
                                    prompt=prompts[int(selected[0])],
                                )
                                premature_close_episodes += 1
                        print(
                            f"added premature-close corrective states {dagger_path}::{name}: "
                            f"{len(premature)} frames x{args.premature_close_repeat}",
                            flush=True,
                        )

                pick_indices = np.flatnonzero(
                    np.asarray([phase in ("PICK_ABOVE", "PICK_DESCEND") for phase in phases])
                )[: args.pick_approach_max_frames]
                if len(pick_indices) and args.pick_approach_repeat:
                    starts = [0] + [
                        i
                        for i in range(1, len(pick_indices))
                        if pick_indices[i] != pick_indices[i - 1] + 1
                        or prompts[pick_indices[i]] != prompts[pick_indices[i - 1]]
                    ]
                    ends = starts[1:] + [len(pick_indices)]
                    for _ in range(args.pick_approach_repeat):
                        for first, last in zip(starts, ends, strict=True):
                            selected = pick_indices[first:last]
                            pick_approach_frames += add_episode(
                                dataset,
                                images=images[selected],
                                wrist_images=wrist_images[selected],
                                states=states[selected],
                                actions=actions[selected],
                                prompt=prompts[int(selected[0])],
                            )
                            pick_approach_episodes += 1
                    print(
                        f"added PICK corrective states {dagger_path}::{name}: "
                        f"{len(pick_indices)} frames x{args.pick_approach_repeat}",
                        flush=True,
                    )

                if args.handle_release_repeat or args.pick_grasp_repeat or args.place_release_repeat:
                    if "model_raw_actions" not in episode:
                        continue  # Older failure files did not preserve unfiltered model actions.
                    model_raw = np.asarray(episode["model_raw_actions"][:limit], dtype=np.float32)
                    if model_raw.shape != (limit, 7):
                        raise ValueError(f"malformed model actions in {dagger_path}::{name}")
                    release = np.flatnonzero(
                        np.isin(phases, ("HANDLE_RELEASE", "HANDLE_RETREAT"))
                        & (actions[:, 6] > 0)
                        & (model_raw[:, 6] < 0)
                    )[: args.handle_release_max_frames]
                    episodes_added, frames_added = add_focused_runs(
                        dataset,
                        selected=release,
                        images=images,
                        wrist_images=wrist_images,
                        states=states,
                        actions=actions,
                        prompts=prompts,
                        repeat=args.handle_release_repeat,
                    )
                    handle_release_episodes += episodes_added
                    handle_release_frames += frames_added
                    if frames_added:
                        print(
                            f"added HANDLE release corrections {dagger_path}::{name}: "
                            f"{len(release)} frames x{args.handle_release_repeat}",
                            flush=True,
                        )

                    grasp = np.flatnonzero(
                        np.isin(phases, ("PICK_CLOSE", "PICK_LIFT")) & (actions[:, 6] < 0) & (model_raw[:, 6] >= 0)
                    )[: args.pick_grasp_max_frames]
                    episodes_added, frames_added = add_focused_runs(
                        dataset,
                        selected=grasp,
                        images=images,
                        wrist_images=wrist_images,
                        states=states,
                        actions=actions,
                        prompts=prompts,
                        repeat=args.pick_grasp_repeat,
                    )
                    pick_grasp_episodes += episodes_added
                    pick_grasp_frames += frames_added
                    if frames_added:
                        print(
                            f"added PICK grasp corrections {dagger_path}::{name}: "
                            f"{len(grasp)} frames x{args.pick_grasp_repeat}",
                            flush=True,
                        )

                    place_release = np.flatnonzero(
                        np.isin(phases, ("PLACE_RELEASE", "PLACE_RETREAT"))
                        & (actions[:, 6] > 0)
                        & (model_raw[:, 6] < 0)
                    )[: args.place_release_max_frames]
                    episodes_added, frames_added = add_focused_runs(
                        dataset,
                        selected=place_release,
                        images=images,
                        wrist_images=wrist_images,
                        states=states,
                        actions=actions,
                        prompts=prompts,
                        repeat=args.place_release_repeat,
                    )
                    place_release_episodes += episodes_added
                    place_release_frames += frames_added
                    if frames_added:
                        print(
                            f"added PLACE release corrections {dagger_path}::{name}: "
                            f"{len(place_release)} frames x{args.place_release_repeat}",
                            flush=True,
                        )

    provenance = {
        "training_only": True,
        "nominal_source": str(args.nominal.resolve()),
        "dagger_sources": [str(path.resolve()) for path in args.dagger],
        "dagger_outcome": "failed",
        "dagger_action_label": "truth_guided_teacher_at_on_policy_observation",
        "state_convention": "joint_pos_rel_joint_vel_rel",
        "dagger_repeat": args.dagger_repeat,
        "max_dagger_frames": args.max_dagger_frames,
        "nominal_episodes": nominal_episodes,
        "nominal_frames": nominal_frames,
        "dagger_episodes": dagger_episodes,
        "dagger_frames": dagger_frames,
        "closed_focus_repeat": args.closed_state_repeat,
        "closed_focus_max_frames_per_failure": args.closed_state_max_frames,
        "closed_focus_episodes": closed_focus_episodes,
        "closed_focus_frames": closed_focus_frames,
        "premature_close_repeat": args.premature_close_repeat,
        "premature_close_max_frames_per_failure": args.premature_close_max_frames,
        "premature_close_episodes": premature_close_episodes,
        "premature_close_frames": premature_close_frames,
        "pick_approach_repeat": args.pick_approach_repeat,
        "pick_approach_max_frames_per_failure": args.pick_approach_max_frames,
        "pick_approach_episodes": pick_approach_episodes,
        "pick_approach_frames": pick_approach_frames,
        "handle_release_repeat": args.handle_release_repeat,
        "handle_release_max_frames_per_failure": args.handle_release_max_frames,
        "handle_release_episodes": handle_release_episodes,
        "handle_release_frames": handle_release_frames,
        "pick_grasp_repeat": args.pick_grasp_repeat,
        "pick_grasp_max_frames_per_failure": args.pick_grasp_max_frames,
        "pick_grasp_episodes": pick_grasp_episodes,
        "pick_grasp_frames": pick_grasp_frames,
        "place_release_repeat": args.place_release_repeat,
        "place_release_max_frames_per_failure": args.place_release_max_frames,
        "place_release_episodes": place_release_episodes,
        "place_release_frames": place_release_frames,
    }
    (output / "training_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
