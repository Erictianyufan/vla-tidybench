"""Validation helpers for a simulation-ready pi0.5 deployment bundle."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vla_tidybench.task_metrics import FORMAL_SUCCESS_HOLD_STEPS, SUCCESS_PREDICATE_VERSION

REQUIRED_CHECKPOINT_FILES = (
    "_CHECKPOINT_METADATA",
    "params/_METADATA",
    "params/manifest.ocdbt",
)
POLICY_MODES = frozenset(("lora", "expert", "full"))
CHECKPOINT_DIGEST_ALGORITHM = "sha256-tree-v1"
FORMAL_SKILLS = ("open", "pick", "place", "close")
FORMAL_MIN_EPISODES_PER_SKILL = 5
FORMAL_MIN_SUCCESS_RATE = 0.6
FORMAL_MAX_P95_INFER_MS = 250.0


@dataclass(frozen=True)
class Deployment:
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    policy_mode: str
    manifest: dict[str, Any]
    evaluation: dict[str, Any] | None


def checkpoint_inventory(checkpoint: Path) -> tuple[int, int]:
    files = _checkpoint_files(checkpoint)
    return len(files), sum(path.stat().st_size for path in files)


def checkpoint_asset_id(checkpoint: Path) -> str:
    """Return the unique normalization asset ID embedded in a checkpoint."""

    assets = checkpoint.expanduser().resolve() / "assets"
    if not assets.is_dir() or assets.is_symlink():
        raise ValueError(f"checkpoint has no real assets directory: {assets}")
    entries = list(assets.rglob("*"))
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise ValueError(f"checkpoint assets contain symbolic links: {symlinks[0]}")
    norm_stats = [path for path in entries if path.is_file() and path.name == "norm_stats.json"]
    if len(norm_stats) != 1:
        raise ValueError(
            f"checkpoint must contain exactly one norm_stats.json, found {len(norm_stats)} under {assets}"
        )
    asset_id = norm_stats[0].parent.relative_to(assets).as_posix()
    if not asset_id or asset_id == ".":
        raise ValueError("checkpoint normalization assets must be stored under a non-empty asset ID")
    return asset_id


def _checkpoint_files(checkpoint: Path) -> list[Path]:
    checkpoint = checkpoint.expanduser().resolve()
    entries = list(checkpoint.rglob("*"))
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise ValueError(f"checkpoint contains symbolic links: {symlinks[0]}")
    files = sorted(
        (path for path in entries if path.is_file()),
        key=lambda path: path.relative_to(checkpoint).as_posix(),
    )
    return files


def checkpoint_fingerprint(checkpoint: Path) -> tuple[int, int, str]:
    """Hash every checkpoint byte plus framed relative paths and file sizes."""

    checkpoint = checkpoint.expanduser().resolve()
    files = _checkpoint_files(checkpoint)
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(checkpoint).as_posix().encode("utf-8")
        size = path.stat().st_size
        total_bytes += size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        bytes_read = 0
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size:
            raise ValueError(f"checkpoint file changed while hashing: {path}")
    return len(files), total_bytes, digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"formal evaluation has invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"formal evaluation has non-finite {label}")
    return result


def validate_formal_evaluation(
    evaluation: dict[str, Any], *, checkpoint: Path, checkpoint_sha256: str
) -> None:
    """Validate a self-consistent, locked four-skill evaluation report."""

    if int(evaluation.get("schema_version", -1)) != 1:
        raise ValueError("unsupported evaluation report schema")
    if not bool(evaluation.get("gate_passed", False)):
        raise ValueError("deployment evaluation gate did not pass")
    if not bool(evaluation.get("autonomous_only", False)):
        raise ValueError("deployment evaluation contains assisted rollouts")
    if evaluation.get("policy") != "pi0.5-drawer-full":
        raise ValueError("formal evaluation does not identify the full drawer policy")
    evaluated_checkpoint = Path(str(evaluation.get("checkpoint", ""))).expanduser().resolve()
    if evaluated_checkpoint != checkpoint.expanduser().resolve():
        raise ValueError(
            f"evaluation checkpoint {evaluated_checkpoint} does not match deployment checkpoint {checkpoint}"
        )
    if evaluation.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("evaluation checkpoint SHA-256 does not match deployment checkpoint")

    required_skills = evaluation.get("required_skills")
    if not isinstance(required_skills, list) or tuple(required_skills) != FORMAL_SKILLS:
        raise ValueError(f"formal evaluation requires skills in locked order {FORMAL_SKILLS}")
    thresholds = evaluation.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("formal evaluation has no thresholds object")
    if int(thresholds.get("min_episodes_per_skill", -1)) < FORMAL_MIN_EPISODES_PER_SKILL:
        raise ValueError("formal evaluation requires at least five episodes per skill")
    if _number(thresholds.get("min_success_rate"), "minimum success rate") < FORMAL_MIN_SUCCESS_RATE:
        raise ValueError("formal evaluation minimum success-rate threshold is too weak")
    if _number(thresholds.get("max_p95_infer_ms"), "maximum P95 latency") > FORMAL_MAX_P95_INFER_MS:
        raise ValueError("formal evaluation P95 latency threshold is too weak")

    episodes = evaluation.get("episodes")
    per_skill = evaluation.get("per_skill")
    if not isinstance(episodes, list) or not isinstance(per_skill, dict):
        raise ValueError("formal evaluation is missing episode or per-skill evidence")
    episode_count = int(evaluation.get("episode_count", -1))
    if episode_count != len(episodes):
        raise ValueError("formal evaluation episode_count disagrees with episode evidence")

    audited_successes = 0
    for skill in FORMAL_SKILLS:
        selected = [episode for episode in episodes if isinstance(episode, dict) and episode.get("skill") == skill]
        if len(selected) < FORMAL_MIN_EPISODES_PER_SKILL:
            raise ValueError(f"formal evaluation skill {skill} has fewer than five episodes")
        seeds = [episode.get("seed") for episode in selected]
        contexts = [episode.get("context") for episode in selected]
        if len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) for seed in seeds):
            raise ValueError(f"formal evaluation skill {skill} has invalid or duplicate seeds")
        invalid_contexts = len(set(contexts)) != len(contexts) or any(
            not isinstance(context, str) or not context for context in contexts
        )
        if invalid_contexts:
            raise ValueError(f"formal evaluation skill {skill} has invalid or duplicate contexts")
        if any(episode.get("checkpoint_sha256") != checkpoint_sha256 for episode in selected):
            raise ValueError(f"formal evaluation skill {skill} mixes checkpoint content")
        if any(episode.get("success_predicate") != SUCCESS_PREDICATE_VERSION for episode in selected):
            raise ValueError(f"formal evaluation skill {skill} mixes success predicates")
        invalid_hold = any(
            (
                bool(episode.get("success", False))
                and int(episode.get("success_hold_steps", -1)) < FORMAL_SUCCESS_HOLD_STEPS
            )
            or (
                not bool(episode.get("success", False))
                and int(episode.get("success_hold_steps", -1)) >= FORMAL_SUCCESS_HOLD_STEPS
            )
            for episode in selected
        )
        if invalid_hold:
            raise ValueError(f"formal evaluation skill {skill} has inconsistent success-hold evidence")

        successes = sum(bool(episode.get("success", False)) for episode in selected)
        audited_successes += successes
        success_rate = successes / len(selected)
        if success_rate < FORMAL_MIN_SUCCESS_RATE:
            raise ValueError(f"formal evaluation skill {skill} success rate is below {FORMAL_MIN_SUCCESS_RATE:.1f}")
        summary = per_skill.get(skill)
        if not isinstance(summary, dict):
            raise ValueError(f"formal evaluation has no per-skill summary for {skill}")
        if int(summary.get("episodes", -1)) != len(selected) or int(summary.get("successes", -1)) != successes:
            raise ValueError(f"formal evaluation per-skill counts disagree for {skill}")
        if not math.isclose(_number(summary.get("success_rate"), f"{skill} success rate"), success_rate):
            raise ValueError(f"formal evaluation per-skill success rate disagrees for {skill}")

    if episode_count != sum(int(per_skill[skill]["episodes"]) for skill in FORMAL_SKILLS):
        raise ValueError("formal evaluation total episode count disagrees with per-skill summaries")
    if int(evaluation.get("successes", -1)) != audited_successes:
        raise ValueError("formal evaluation total success count disagrees with episode evidence")
    overall_rate = audited_successes / episode_count if episode_count else 0.0
    if not math.isclose(_number(evaluation.get("overall_success_rate"), "overall success rate"), overall_rate):
        raise ValueError("formal evaluation overall success rate disagrees with episode evidence")
    if _number(evaluation.get("p95_infer_ms"), "P95 inference latency") > FORMAL_MAX_P95_INFER_MS:
        raise ValueError("formal evaluation P95 inference latency exceeds the deployment limit")


def load_deployment(path: Path, *, require_validated: bool = True) -> Deployment:
    """Load and verify an exported deployment before policy construction.

    Formal bundles must be clean-code exports with a checksum-bound autonomous
    evaluation report. Systems-smoke bundles can be inspected by opting out of
    the formal evaluation requirement explicitly.
    """

    root = path.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, "deployment manifest")
    format_version = int(manifest.get("format_version", -1))
    if format_version not in (1, 2):
        raise ValueError(f"unsupported deployment format_version in {manifest_path}")
    if require_validated and format_version != 2:
        raise ValueError("formal deployment requires content-hashed format_version 2")

    policy_mode = str(manifest.get("policy_mode", ""))
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"unsupported deployment policy_mode: {policy_mode!r}")
    if require_validated:
        if policy_mode != "full" or manifest.get("policy_config") != "drawer_four_skill":
            raise ValueError("formal deployment requires the full drawer_four_skill policy")
        if manifest.get("stage") != "stage3-hard-recovery":
            raise ValueError("formal deployment must come from stage3-hard-recovery")
        if not str(manifest.get("dataset_repo", "")).strip():
            raise ValueError("formal deployment does not identify its dataset repository")
        project_commit = str(manifest.get("project_commit", ""))
        valid_commit = len(project_commit) in (40, 64) and all(
            char in "0123456789abcdef" for char in project_commit
        )
        if not valid_commit:
            raise ValueError("formal deployment has no valid project commit")
        if bool(manifest.get("project_dirty", True)):
            raise ValueError("formal deployment was exported from a dirty project checkout")

    checkpoint_entry = root / "checkpoint"
    checkpoint_storage = str(manifest.get("checkpoint_storage", "symlink"))
    recorded_checkpoint = Path(str(manifest.get("checkpoint", ""))).expanduser().resolve()
    if checkpoint_storage == "symlink":
        if not checkpoint_entry.is_symlink():
            raise ValueError(f"deployment checkpoint must be a symbolic link: {checkpoint_entry}")
        checkpoint = checkpoint_entry.resolve(strict=True)
        if checkpoint != recorded_checkpoint:
            raise ValueError(
                f"checkpoint link resolves to {checkpoint}, manifest records {recorded_checkpoint}"
            )
    elif checkpoint_storage == "copy":
        if checkpoint_entry.is_symlink() or not checkpoint_entry.is_dir():
            raise ValueError(f"copied deployment checkpoint must be a real directory: {checkpoint_entry}")
        checkpoint = checkpoint_entry.resolve(strict=True)
        if checkpoint.parent != root:
            raise ValueError(f"copied deployment checkpoint escapes deployment root: {checkpoint}")
    else:
        raise ValueError(f"unsupported deployment checkpoint_storage: {checkpoint_storage!r}")
    missing = [
        str(checkpoint / relative)
        for relative in REQUIRED_CHECKPOINT_FILES
        if not (checkpoint / relative).is_file()
    ]
    if missing:
        raise ValueError("deployment checkpoint is incomplete; missing: " + ", ".join(missing))
    embedded_asset_id = checkpoint_asset_id(checkpoint)
    if require_validated and manifest.get("dataset_repo") != embedded_asset_id:
        raise ValueError(
            "formal deployment dataset_repo does not match checkpoint normalization asset ID: "
            f"{manifest.get('dataset_repo')!r} != {embedded_asset_id!r}"
        )

    file_count, byte_count, checkpoint_sha256 = checkpoint_fingerprint(checkpoint)
    if file_count != int(manifest.get("file_count", -1)):
        raise ValueError(f"checkpoint file count changed: expected {manifest.get('file_count')}, got {file_count}")
    if byte_count != int(manifest.get("byte_count", -1)):
        raise ValueError(f"checkpoint byte count changed: expected {manifest.get('byte_count')}, got {byte_count}")
    checkpoint_digest = manifest.get("checkpoint_digest")
    if format_version == 2:
        if not isinstance(checkpoint_digest, dict):
            raise ValueError("deployment checkpoint_digest must be an object")
        if checkpoint_digest.get("algorithm") != CHECKPOINT_DIGEST_ALGORITHM:
            raise ValueError("unsupported deployment checkpoint digest algorithm")
        if checkpoint_digest.get("sha256") != checkpoint_sha256:
            raise ValueError("checkpoint content SHA-256 does not match deployment manifest")

    evaluation_manifest = manifest.get("evaluation")
    evaluation: dict[str, Any] | None = None
    if evaluation_manifest is None:
        if require_validated:
            raise ValueError("formal deployment has no evaluation record")
    else:
        if not isinstance(evaluation_manifest, dict):
            raise ValueError("deployment evaluation entry must be an object")
        evaluation_path = root / "evaluation.json"
        evaluation = _load_json(evaluation_path, "evaluation")
        digest = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        if digest != evaluation_manifest.get("sha256"):
            raise ValueError("evaluation checksum does not match deployment manifest")
        if format_version == 2:
            validate_formal_evaluation(
                evaluation,
                checkpoint=recorded_checkpoint,
                checkpoint_sha256=checkpoint_sha256,
            )

    return Deployment(root, checkpoint, checkpoint_sha256, policy_mode, manifest, evaluation)
