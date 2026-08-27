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
    training: dict[str, Any] | None
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


def validate_training_completion(
    training: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    file_count: int,
    byte_count: int,
    dataset_repo: str,
    require_clean_provenance: bool = True,
) -> None:
    """Validate a content-bound completion report for one formal training stage."""

    if int(training.get("schema_version", -1)) != 1 or training.get("verified") is not True:
        raise ValueError("formal deployment has no verified training completion report")
    checkpoint = checkpoint.expanduser().resolve()
    recorded_checkpoint = Path(str(training.get("checkpoint", ""))).expanduser().resolve()
    if recorded_checkpoint != checkpoint:
        raise ValueError("training completion checkpoint does not match deployment checkpoint")
    try:
        final_step = int(training.get("final_step", -1))
        num_train_steps = int(training.get("num_train_steps", -1))
        recorded_files = int(training.get("checkpoint_file_count", -1))
        recorded_bytes = int(training.get("checkpoint_byte_count", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("training completion has invalid numeric identities") from error
    if not checkpoint.name.isdigit() or final_step != int(checkpoint.name):
        raise ValueError("training completion final step does not match its checkpoint")
    if num_train_steps != final_step + 1:
        raise ValueError("training completion step count is inconsistent")
    if training.get("dataset_repo") != dataset_repo:
        raise ValueError("training completion dataset does not match deployment dataset")
    if training.get("checkpoint_digest_algorithm") != CHECKPOINT_DIGEST_ALGORITHM:
        raise ValueError("training completion uses an unsupported checkpoint digest")
    if training.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("training completion checkpoint SHA-256 does not match deployed content")
    if recorded_files != file_count or recorded_bytes != byte_count:
        raise ValueError("training completion checkpoint inventory does not match deployed content")
    if require_clean_provenance:
        project_commit = str(training.get("project_commit", ""))
        valid_commit = len(project_commit) in (40, 64) and all(
            char in "0123456789abcdef" for char in project_commit
        )
        if not valid_commit or training.get("project_dirty") is not False:
            raise ValueError("training completion requires an exact clean project commit")
        for label in ("openpi_source", "init_params"):
            digest = str(training.get(f"{label}_sha256", ""))
            try:
                count = int(training.get(f"{label}_files", 0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"training completion has invalid {label} fingerprint") from error
            if (
                len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or count < 1
            ):
                raise ValueError(f"training completion has invalid {label} fingerprint")
        dataset_digest = str(training.get("dataset_sha256", ""))
        try:
            dataset_files = int(training.get("dataset_files", 0))
            dataset_bytes = int(training.get("dataset_bytes", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("training completion has an invalid dataset fingerprint") from error
        if (
            not str(training.get("dataset_path", "")).strip()
            or training.get("dataset_digest_algorithm") != CHECKPOINT_DIGEST_ALGORITHM
            or len(dataset_digest) != 64
            or any(char not in "0123456789abcdef" for char in dataset_digest)
            or dataset_files < 1
            or dataset_bytes < 1
        ):
            raise ValueError("training completion has an invalid dataset fingerprint")
    for label in ("loss", "grad_norm", "param_norm"):
        value = _number(training.get(label), f"training {label}")
        if value < 0:
            raise ValueError(f"training completion has negative {label}")


def validate_formal_evaluation(
    evaluation: dict[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    project_commit: str | None = None,
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
    evaluated_commit = str(evaluation.get("project_commit", ""))
    valid_commit = len(evaluated_commit) in (40, 64) and all(
        char in "0123456789abcdef" for char in evaluated_commit
    )
    if not valid_commit:
        raise ValueError("formal evaluation has no valid project commit")
    if project_commit is not None and evaluated_commit != project_commit:
        raise ValueError("formal evaluation project commit does not match deployment code")

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

    context_lock = evaluation.get("context_lock")
    if not isinstance(context_lock, dict) or int(context_lock.get("schema_version", -1)) != 1:
        raise ValueError("formal evaluation has no valid context lock")
    encoded_lock = (json.dumps(context_lock, indent=2) + "\n").encode()
    if hashlib.sha256(encoded_lock).hexdigest() != evaluation.get("context_lock_sha256"):
        raise ValueError("formal evaluation context-lock SHA-256 is invalid")
    manifest_digest = str(context_lock.get("context_manifest_sha256", ""))
    if context_lock.get("context_manifest") != "main_validation.json" or len(manifest_digest) != 64 or any(
        char not in "0123456789abcdef" for char in manifest_digest
    ):
        raise ValueError("formal evaluation context lock has invalid manifest identity")
    context_sources = context_lock.get("sources")
    if not isinstance(context_sources, list):
        raise ValueError("formal evaluation context lock has no sources")
    expected_context_files = {f"drawer_{skill}_formal.hdf5" for skill in FORMAL_SKILLS}
    locked_contexts: dict[str, set[str]] = {}
    locked_bytes = 0
    locked_count = 0
    for source in context_sources:
        if not isinstance(source, dict):
            raise ValueError("formal evaluation context-lock sources must be objects")
        file = str(source.get("file", ""))
        digest = str(source.get("sha256", ""))
        if file in locked_contexts or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("formal evaluation context lock has invalid source identity")
        names = source.get("episode_names")
        indices = source.get("episode_indices")
        source_bytes = int(source.get("bytes", -1))
        if (
            not isinstance(names, list)
            or not isinstance(indices, list)
            or len(names) != len(indices)
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or index < 0 for index in indices)
            or source_bytes <= 0
        ):
            raise ValueError(f"formal evaluation context lock has invalid episodes for {file}")
        locked_contexts[file] = set(names)
        locked_count += len(names)
        locked_bytes += source_bytes
    if set(locked_contexts) != expected_context_files:
        raise ValueError("formal evaluation context lock does not contain the four skill sources")
    if int(context_lock.get("context_count", -1)) != locked_count:
        raise ValueError("formal evaluation context-lock episode count is inconsistent")
    if int(context_lock.get("total_bytes", -1)) != locked_bytes or locked_bytes <= 0:
        raise ValueError("formal evaluation context-lock byte count is inconsistent")
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("formal evaluation episode evidence must contain objects")
        try:
            context_file, context_episode = str(episode.get("context", "")).split("::", maxsplit=1)
        except ValueError as error:
            raise ValueError("formal evaluation episode has malformed context identity") from error
        if context_episode not in locked_contexts.get(context_file, set()):
            raise ValueError(
                f"formal evaluation episode context is not content-locked: {context_file}::{context_episode}"
            )

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
        if any(episode.get("project_commit") != evaluated_commit for episode in selected):
            raise ValueError(f"formal evaluation skill {skill} mixes project commits")
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
    if format_version not in (1, 2, 3):
        raise ValueError(f"unsupported deployment format_version in {manifest_path}")
    if require_validated and format_version != 3:
        raise ValueError("formal deployment requires provenance-bound format_version 3")

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
    if format_version >= 2:
        if not isinstance(checkpoint_digest, dict):
            raise ValueError("deployment checkpoint_digest must be an object")
        if checkpoint_digest.get("algorithm") != CHECKPOINT_DIGEST_ALGORITHM:
            raise ValueError("unsupported deployment checkpoint digest algorithm")
        if checkpoint_digest.get("sha256") != checkpoint_sha256:
            raise ValueError("checkpoint content SHA-256 does not match deployment manifest")

    training_manifest = manifest.get("training")
    training: dict[str, Any] | None = None
    if training_manifest is None:
        if require_validated:
            raise ValueError("formal deployment has no training completion record")
    else:
        if not isinstance(training_manifest, dict):
            raise ValueError("deployment training entry must be an object")
        if training_manifest.get("path") != "training_completion.json":
            raise ValueError("deployment training entry has an invalid path")
        training_path = root / "training_completion.json"
        training = _load_json(training_path, "training completion")
        digest = hashlib.sha256(training_path.read_bytes()).hexdigest()
        if digest != training_manifest.get("sha256"):
            raise ValueError("training completion checksum does not match deployment manifest")
        validate_training_completion(
            training,
            checkpoint=recorded_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            file_count=file_count,
            byte_count=byte_count,
            dataset_repo=str(manifest.get("dataset_repo", "")),
            require_clean_provenance=require_validated,
        )
        for key in (
            "project_commit",
            "openpi_source_sha256",
            "init_params_sha256",
            "dataset_digest_algorithm",
            "dataset_sha256",
        ):
            if training_manifest.get(key) != training.get(key):
                raise ValueError(f"deployment training identity disagrees for {key}")

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
        if format_version >= 2:
            validate_formal_evaluation(
                evaluation,
                checkpoint=recorded_checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                project_commit=str(manifest.get("project_commit", "")),
            )

    return Deployment(root, checkpoint, checkpoint_sha256, policy_mode, manifest, training, evaluation)
