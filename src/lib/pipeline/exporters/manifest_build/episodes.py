"""Episode preparation and feature loading for manifest-based build/export."""

from __future__ import annotations

import io
import json
import tarfile
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

from lib.pipeline.clips.annotation_protocol import build_annotation_issue_from_candidates, load_clip_annotation
from lib.pipeline.clips.clip_manifest import ClipManifestRecord, load_clip_manifest
from lib.pipeline.slam.depth_artifacts import load_export_depths
from lib.pipeline.hands.hand_depth_align import HandDepthAlignConfig
from lib.pipeline.io.frame_sources import build_frame_bytes_reader, classify_descriptor_storage, validate_descriptor_for_frame_reads
from lib.pipeline.exporters.mano_codec import build_mano_pca_frame_features
from lib.pipeline.exporters.webdataset_features import (
    InvalidCameraDataError,
    _build_lowdim_features,
    _compute_joint_states,
    _compute_presence_per_frame,
    _load_episode_camera_features,
    _load_world_space_prediction,
    export_frame_count_with_action,
)
from lib.pipeline.quality.quality_metrics import (
    finalize_clip_quality_metrics,
    new_clip_quality_stats,
    parse_instruction_metadata,
    update_clip_quality_stats,
    validate_lowdim_numeric_sanity,
)

from .cache import load_cached_features, write_cached_features
from .resample import resample_episode_features


NATIVE_FEATURE_SOURCE = "wds_lowdim_mano_v1"
NATIVE_LOWDIM_SHAPE = (116,)
NATIVE_MANO_SHAPE = (2, 55)
NATIVE_WRIST_STATE_SLICE = slice(0, 18)
NATIVE_HAND_STATE_SLICE = slice(18, 48)
NATIVE_WRIST_ACTION_SLICE = slice(48, 66)
NATIVE_HAND_ACTION_SLICE = slice(66, 96)
NATIVE_EXTRINSIC_SLICE = slice(96, 112)
NATIVE_INTRINSIC_SLICE = slice(112, 116)


def descriptor_uses_native_features(descriptor) -> bool:
    return (descriptor.extra or {}).get("native_feature_source") == NATIVE_FEATURE_SOURCE


def _native_member_name_from_image(frame_name: str, suffix: str) -> str:
    image_suffix = ".image.jpg"
    if not str(frame_name).endswith(image_suffix):
        raise ValueError(f"Native WDS frame name must end with {image_suffix}: {frame_name}")
    return f"{str(frame_name)[:-len(image_suffix)]}{suffix}"


def _decode_native_npy(payload: bytes, *, sample_key: str, field_name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as error:
        raise ValueError(f"Failed to decode native {field_name} for {sample_key}: {error}") from error
    array = np.asarray(array, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"Native {field_name} shape mismatch for {sample_key}: expected={expected_shape}, got={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Native {field_name} contains non-finite values for {sample_key}")
    return array


def _decode_native_presence(payload: bytes, *, sample_key: str) -> int:
    try:
        meta = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise ValueError(f"Failed to decode native meta for {sample_key}: {error}") from error
    try:
        presence = int(meta.get("presence", 0))
    except Exception as error:
        raise ValueError(f"Invalid native presence for {sample_key}: {meta.get('presence')!r}") from error
    if presence < 0 or presence > 3:
        raise ValueError(f"Invalid native presence for {sample_key}: {presence}")
    return presence


def _validate_native_lowdim(lowdim_all: np.ndarray) -> None:
    extrinsics = lowdim_all[:, NATIVE_EXTRINSIC_SLICE].reshape(-1, 4, 4)
    intrinsics = lowdim_all[:, NATIVE_INTRINSIC_SLICE]
    if not np.isfinite(extrinsics).all():
        raise ValueError("Native lowdim extrinsics contain non-finite values")
    if not np.isfinite(intrinsics).all():
        raise ValueError("Native lowdim intrinsics contain non-finite values")
    if (intrinsics[:, 0] <= 0).any() or (intrinsics[:, 1] <= 0).any():
        raise ValueError("Native lowdim intrinsics contain non-positive focal length")
    expected_bottom_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(extrinsics[:, 3, :], expected_bottom_row[None, :], atol=1e-3):
        raise ValueError("Native lowdim extrinsics are not homogeneous World2Cam matrices")
    rotation_dets = np.linalg.det(extrinsics[:, :3, :3].astype(np.float64))
    if not np.isfinite(rotation_dets).all() or (np.abs(rotation_dets) < 1e-6).any():
        raise ValueError("Native lowdim extrinsics contain singular camera rotations")
    for frame_idx, lowdim in enumerate(lowdim_all):
        sanity = validate_lowdim_numeric_sanity(lowdim)
        if not sanity["valid"]:
            raise ValueError(
                f"Native lowdim frame {frame_idx} failed numeric sanity: {','.join(sanity['issues'])}"
            )
    if lowdim_all.shape[0] > 1:
        wrist_action = lowdim_all[:-1, NATIVE_WRIST_ACTION_SLICE]
        next_wrist_state = lowdim_all[1:, NATIVE_WRIST_STATE_SLICE]
        hand_action = lowdim_all[:-1, NATIVE_HAND_ACTION_SLICE]
        next_hand_state = lowdim_all[1:, NATIVE_HAND_STATE_SLICE]
        if not np.allclose(wrist_action, next_wrist_state, atol=1e-4, rtol=1e-4):
            raise ValueError("Native lowdim wrist_action does not match next-frame wrist_state")
        if not np.allclose(hand_action, next_hand_state, atol=1e-4, rtol=1e-4):
            raise ValueError("Native lowdim hand_action does not match next-frame hand_state")


def _load_native_descriptor_episode_features(ep: dict) -> dict | None:
    descriptor = ep["descriptor"]
    if not descriptor_uses_native_features(descriptor):
        return None
    if not descriptor.shard_path:
        raise ValueError(f"Native feature descriptor {descriptor.clip_id} missing shard_path")

    requested_frame_count = int(ep.get("num_valid_frames") or descriptor.frame_count)
    frame_names = list(descriptor.frame_names[:requested_frame_count])
    if not frame_names:
        return None

    wanted: dict[str, tuple[int, str, str]] = {}
    sample_keys = []
    for frame_idx, frame_name in enumerate(frame_names):
        sample_key = str(frame_name)[: -len(".image.jpg")]
        sample_keys.append(sample_key)
        wanted[_native_member_name_from_image(frame_name, ".lowdim.npy")] = (frame_idx, "lowdim", sample_key)
        wanted[_native_member_name_from_image(frame_name, ".mano.npy")] = (frame_idx, "mano", sample_key)
        wanted[_native_member_name_from_image(frame_name, ".meta.json")] = (frame_idx, "meta", sample_key)

    lowdim_all = np.empty((len(frame_names), 116), dtype=np.float32)
    mano_all = np.empty((len(frame_names), 2, 55), dtype=np.float32)
    presence_per_frame = np.zeros((len(frame_names),), dtype=np.uint8)
    seen = set()

    with tarfile.open(descriptor.shard_path, "r|") as tar_reader:
        for member in tar_reader:
            if not member.isfile() or member.name not in wanted:
                continue
            frame_idx, field_name, sample_key = wanted[member.name]
            member_file = tar_reader.extractfile(member)
            if member_file is None:
                raise ValueError(f"Failed to extract native {field_name}: {member.name}")
            payload = member_file.read()
            if field_name == "lowdim":
                lowdim_all[frame_idx] = _decode_native_npy(
                    payload,
                    sample_key=sample_key,
                    field_name=field_name,
                    expected_shape=NATIVE_LOWDIM_SHAPE,
                )
            elif field_name == "mano":
                mano_all[frame_idx] = _decode_native_npy(
                    payload,
                    sample_key=sample_key,
                    field_name=field_name,
                    expected_shape=NATIVE_MANO_SHAPE,
                )
            else:
                presence_per_frame[frame_idx] = _decode_native_presence(payload, sample_key=sample_key)
            seen.add(member.name)

    missing = sorted(set(wanted) - seen)
    if missing:
        raise ValueError(f"Native WDS feature payloads missing for {descriptor.clip_id}: {missing[:8]}")
    _validate_native_lowdim(lowdim_all)
    return {
        "frame_count": len(frame_names),
        "lowdim_all": lowdim_all,
        "mano_all": mano_all,
        "presence_per_frame": presence_per_frame,
    }


def load_descriptor_episode_features(
    ep: dict,
    mano_right,
    mano_left,
    device,
    feature_cache_dir: str | None,
    mano_dir: str | None,
    *,
    prediction: dict | None = None,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
    export_depth: bool = False,
):
    seq_folder = ep["seq_folder"]
    descriptor = ep.get("descriptor")
    requested_export_frame_count = ep.get("num_valid_frames")
    if requested_export_frame_count is None and "frame_end" in ep:
        requested_export_frame_count = int(ep["frame_end"] - ep.get("frame_start", 0))
    if requested_export_frame_count is not None:
        requested_export_frame_count = int(requested_export_frame_count)
        cached = (
            load_cached_features(
                seq_folder,
                requested_export_frame_count,
                feature_cache_dir,
                source_fps=source_fps,
                target_fps=target_fps,
                interpolate_labels=interpolate_labels,
            )
            if feature_cache_dir
            else None
        )
        if cached is not None:
            if export_depth:
                try:
                    cached["depth_all"] = load_export_depths(seq_folder, int(cached["frame_count"]))
                except Exception as error:
                    print(f"  Skip {ep['episode_id']}: invalid depth artifact: {error}")
                    return None
            return cached

    if descriptor is not None and descriptor_uses_native_features(descriptor):
        try:
            episode_data = _load_native_descriptor_episode_features(ep)
        except Exception as error:
            print(f"  Skip {ep['episode_id']}: invalid native WDS features: {error}")
            return None
        if episode_data is None:
            return None
        if export_depth:
            try:
                episode_data["depth_all"] = load_export_depths(seq_folder, int(episode_data["frame_count"]))
            except Exception as error:
                print(f"  Skip {ep['episode_id']}: invalid depth artifact: {error}")
                return None
        write_cached_features(
            seq_folder,
            feature_cache_dir,
            episode_data,
            source_fps=source_fps,
            target_fps=target_fps,
            interpolate_labels=interpolate_labels,
        )
        return episode_data

    if prediction is None:
        prediction = _load_world_space_prediction({"episode_id": ep["episode_id"]}, str(Path(seq_folder) / "world_space_res.pth"))
    if prediction is None:
        return None

    pred_trans = prediction["pred_trans"]
    pred_rot = prediction["pred_rot"]
    pred_hand_pose = prediction["pred_hand_pose"]
    pred_betas = prediction["pred_betas"]
    pred_valid = prediction["pred_valid"]
    source_frame_count = int(pred_trans.shape[1])
    if requested_export_frame_count is None:
        if interpolate_labels and source_fps > 0 and target_fps > 0 and source_frame_count > 1:
            duration = float(source_frame_count - 1) / float(source_fps)
            requested_source_frame_count = int(round(duration * float(target_fps))) + 1
        else:
            requested_source_frame_count = source_frame_count
        frame_count = int(
            requested_source_frame_count if interpolate_labels else min(int(requested_source_frame_count), source_frame_count)
        )
        export_frame_count = export_frame_count_with_action(frame_count)
        if export_frame_count <= 0:
            return None
    else:
        # `num_valid_frames` / `frame_end - frame_start` already represent the
        # final exportable frame count after dropping the last frame that lacks
        # next-frame action. Reconstruct one extra source frame here so lowdim
        # action targets can be formed without silently dropping another frame.
        export_frame_count = int(requested_export_frame_count)
        if export_frame_count <= 0:
            return None
        frame_count = int(export_frame_count + 1)
        if not interpolate_labels and source_frame_count < frame_count:
            print(
                f"  Skip {ep['episode_id']}: source prediction shorter than requested export span: "
                f"source={source_frame_count} requested_export={export_frame_count}"
            )
            return None

    cached = (
        load_cached_features(
            seq_folder,
            export_frame_count,
            feature_cache_dir,
            source_fps=source_fps,
            target_fps=target_fps,
            interpolate_labels=interpolate_labels,
        )
        if feature_cache_dir
        else None
    )
    if cached is not None:
        return cached

    camera_ep = {"crop_dir": seq_folder, "episode_id": ep["episode_id"]}
    try:
        # Load camera + presence first so the optional hand-depth alignment can
        # ray-scale joints; keep manifest lowdim consistent with the WDS export.
        extrinsics, intrinsic = _load_episode_camera_features(camera_ep, source_frame_count)
        presence_per_frame = _compute_presence_per_frame(pred_valid, source_frame_count)
        align_ctx = {
            "cfg": HandDepthAlignConfig.from_env(),
            "extrinsics": extrinsics[:source_frame_count],
            "crop_dir": seq_folder,
            "presence": presence_per_frame[:source_frame_count],
        }
        wrist_state, hand_state = _compute_joint_states(
            pred_trans,
            pred_rot,
            pred_hand_pose,
            pred_betas,
            mano_right,
            mano_left,
            device,
            align_ctx=align_ctx,
        )
        wrist_state, hand_state, pred_rot, pred_hand_pose, pred_betas, extrinsics, presence_per_frame = resample_episode_features(
            wrist_state[:source_frame_count],
            hand_state[:source_frame_count],
            pred_rot[:, :source_frame_count],
            pred_hand_pose[:, :source_frame_count],
            pred_betas[:, :source_frame_count],
            extrinsics[:source_frame_count],
            presence_per_frame[:source_frame_count],
            frame_count,
            source_fps=source_fps,
            target_fps=target_fps,
            interpolate_labels=interpolate_labels,
        )
    except InvalidCameraDataError as error:
        print(f"  Skip {ep['episode_id']}: invalid camera features: {error}")
        return None
    lowdim_all = _build_lowdim_features(
        wrist_state,
        hand_state,
        extrinsics[:frame_count],
        intrinsic,
    )
    mano_all = build_mano_pca_frame_features(
        pred_hand_pose[:, :frame_count].cpu().numpy(),
        pred_betas[:, :frame_count].cpu().numpy(),
        mano_dir=mano_dir,
    )

    episode_data = {
        "frame_count": export_frame_count,
        "lowdim_all": lowdim_all[:export_frame_count],
        "mano_all": mano_all[:export_frame_count],
        "presence_per_frame": presence_per_frame[:export_frame_count],
    }
    if export_depth:
        try:
            episode_data["depth_all"] = load_export_depths(seq_folder, int(export_frame_count))
        except Exception as error:
            print(f"  Skip {ep['episode_id']}: invalid depth artifact: {error}")
            return None
    write_cached_features(
        seq_folder,
        feature_cache_dir,
        episode_data,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    return episode_data


def compute_descriptor_episode_quality_metrics(
    ep: dict,
    mano_right,
    mano_left,
    device,
    feature_cache_dir: str | None,
    mano_dir: str | None,
    *,
    prediction: dict | None = None,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
    fatal_offscreen_scale: float = 1.4,
    chunk_window_past_seconds: float = 6.0,
    chunk_window_future_frames: int = 30,
    enable_chunk_window: bool = True,
):
    episode_data = load_descriptor_episode_features(
        ep,
        mano_right,
        mano_left,
        device,
        feature_cache_dir,
        mano_dir,
        prediction=prediction,
        source_fps=source_fps,
        target_fps=target_fps,
        interpolate_labels=interpolate_labels,
    )
    if episode_data is None:
        return None

    frame_count = int(episode_data["frame_count"])
    lowdim_all = np.asarray(episode_data["lowdim_all"])
    presence_per_frame = np.asarray(episode_data["presence_per_frame"])
    if (
        frame_count <= 0
        or lowdim_all.ndim != 2
        or lowdim_all.shape[0] < frame_count
        or presence_per_frame.shape[0] < frame_count
    ):
        return None

    stats = new_clip_quality_stats(
        ep["clip_id"],
        target_fps=float(target_fps),
        chunk_window_past_seconds=float(chunk_window_past_seconds),
        chunk_window_future_frames=int(chunk_window_future_frames),
        enable_chunk_window=bool(enable_chunk_window),
    )
    parsed_instruction = parse_instruction_metadata(ep)
    instruction_num = int(parsed_instruction["instruction_num"])
    image_size = _load_descriptor_image_size(ep["descriptor"])
    for frame_idx in range(frame_count):
        update_clip_quality_stats(
            stats,
            frame_idx,
            instruction_num,
            int(presence_per_frame[frame_idx]),
            lowdim_all[frame_idx],
            missing_instruction=bool(parsed_instruction["missing_instruction"]),
            empty_instruction=bool(parsed_instruction["empty_instruction"]),
            instruction_num_mismatch=bool(parsed_instruction["instruction_num_mismatch"]),
            image_size=image_size,
            severe_offscreen_scale=float(fatal_offscreen_scale),
        )
    return finalize_clip_quality_metrics(stats)


def _load_descriptor_image_size(descriptor) -> tuple[int, int]:
    read_frame_bytes = build_frame_bytes_reader(descriptor)
    first_frame_bytes = read_frame_bytes(0)
    with Image.open(io.BytesIO(first_frame_bytes)) as image:
        width, height = image.size
    return int(width), int(height)


def _prepare_manifest_episode(
    record: ClipManifestRecord,
    require_annotation: bool,
    annotation_root: str | None,
    annotation_suffix: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    try:
        validate_descriptor_for_frame_reads(record.descriptor)
    except Exception:
        return None, "invalid_descriptor", None

    seq_folder = Path(record.descriptor.seq_folder)
    if descriptor_uses_native_features(record.descriptor):
        source_num_frames = int(record.descriptor.frame_count)
        target_num_frames = int(record.descriptor.frame_count)
        num_frames = int(target_num_frames if interpolate_labels else min(source_num_frames, target_num_frames))
        if num_frames <= 0:
            return None, "empty_frames", None
        language = None
        instruction = []
        annotation_issue = None
        if annotation_root:
            annotation, error_code, resolved_path = load_clip_annotation(
                annotation_root,
                record.clip_id,
                annotation_suffix=annotation_suffix,
            )
            if annotation is None:
                annotation_issue = build_annotation_issue_from_candidates(
                    annotation_root,
                    record.clip_id,
                    error_code,
                    annotation_suffix=annotation_suffix,
                    resolved_path=resolved_path,
                )
                if require_annotation:
                    return None, error_code, annotation_issue
            else:
                instruction = annotation.instruction
                language = annotation.language
        return {
            "clip_id": record.clip_id,
            "episode_id": record.clip_id,
            "seq_folder": str(seq_folder),
            "source_id": record.source_id,
            "split": record.split,
            "descriptor": record.descriptor,
            "num_valid_frames": export_frame_count_with_action(num_frames),
            "source_num_frames": source_num_frames,
            "source_fps": float(source_fps),
            "target_fps": float(target_fps),
            "interpolate_labels": bool(interpolate_labels),
            "instruction": instruction,
            "instruction_num": len(instruction),
            "language": language,
        }, None, annotation_issue

    world_res_path = seq_folder / "world_space_res.pth"
    if not world_res_path.exists():
        return None, "missing_world_res", None

    try:
        pred_trans, *_ = joblib.load(world_res_path)
    except Exception:
        return None, "invalid_world_res", None

    source_num_frames = int(np.asarray(pred_trans).shape[1])
    target_num_frames = int(record.descriptor.frame_count)
    num_frames = int(target_num_frames if interpolate_labels else min(source_num_frames, target_num_frames))
    if num_frames <= 0:
        return None, "empty_frames", None

    language = None
    instruction = []
    annotation_issue = None
    if annotation_root:
        annotation, error_code, resolved_path = load_clip_annotation(
            annotation_root,
            record.clip_id,
            annotation_suffix=annotation_suffix,
        )
        if annotation is None:
            annotation_issue = build_annotation_issue_from_candidates(
                annotation_root,
                record.clip_id,
                error_code,
                annotation_suffix=annotation_suffix,
                resolved_path=resolved_path,
            )
            if require_annotation:
                return None, error_code, annotation_issue
        else:
            instruction = annotation.instruction
            language = annotation.language

    return {
        "clip_id": record.clip_id,
        "episode_id": record.clip_id,
        "seq_folder": str(seq_folder),
        "source_id": record.source_id,
        "split": record.split,
        "descriptor": record.descriptor,
        "num_valid_frames": export_frame_count_with_action(num_frames),
        "source_num_frames": source_num_frames,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "interpolate_labels": bool(interpolate_labels),
        "instruction": instruction,
        "instruction_num": len(instruction),
        "language": language,
    }, None, annotation_issue


def prepare_manifest_record_for_build(
    record: ClipManifestRecord,
    *,
    require_annotation: bool,
    annotation_root: str | None,
    annotation_suffix: str,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
    prediction: dict | None = None,
):
    if prediction is None:
        episode, error_code, _annotation_issue = _prepare_manifest_episode(
            record,
            require_annotation,
            annotation_root,
            annotation_suffix,
            source_fps,
            target_fps,
            interpolate_labels,
        )
        return episode, error_code

    try:
        source_num_frames = int(prediction["pred_trans"].shape[1])
    except Exception:
        return None, "invalid_world_res"

    target_num_frames = int(record.descriptor.frame_count)
    num_frames = int(target_num_frames if interpolate_labels else min(source_num_frames, target_num_frames))
    if num_frames <= 0:
        return None, "empty_frames"

    language = None
    instruction = []
    if annotation_root:
        annotation, error_code, _ = load_clip_annotation(
            annotation_root,
            record.clip_id,
            annotation_suffix=annotation_suffix,
        )
        if annotation is None:
            if require_annotation:
                return None, error_code
        else:
            instruction = annotation.instruction
            language = annotation.language

    return {
        "clip_id": record.clip_id,
        "episode_id": record.clip_id,
        "seq_folder": str(Path(record.descriptor.seq_folder)),
        "source_id": record.source_id,
        "split": record.split,
        "descriptor": record.descriptor,
        "num_valid_frames": export_frame_count_with_action(num_frames),
        "source_num_frames": source_num_frames,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "interpolate_labels": bool(interpolate_labels),
        "instruction": instruction,
        "instruction_num": len(instruction),
        "language": language,
    }, None


def load_manifest_record_prediction(record: ClipManifestRecord):
    if descriptor_uses_native_features(record.descriptor):
        return None, "native_features"
    seq_folder = Path(record.descriptor.seq_folder)
    world_res_path = seq_folder / "world_space_res.pth"
    if not world_res_path.exists():
        return None, "missing_world_res"
    prediction = _load_world_space_prediction({"episode_id": record.clip_id}, str(world_res_path))
    if prediction is None:
        return None, "invalid_world_res"
    return prediction, None


def prepare_manifest_episodes(
    manifest_path: str,
    *,
    annotation_root: str | None,
    annotation_suffix: str,
    require_annotation: bool,
    max_episodes: int | None,
    preprocess_workers: int,
    source_fps: float,
    target_fps: float,
    interpolate_labels: bool,
):
    records = load_clip_manifest(manifest_path)
    if max_episodes is not None:
        records = records[:max_episodes]

    stats = {
        "kept": 0,
        "invalid_descriptor": 0,
        "missing_world_res": 0,
        "invalid_world_res": 0,
        "empty_frames": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
        "descriptor_paths": {
            "light_tar": 0,
            "heavy_tar": 0,
            "image_sequence": 0,
        },
        "annotation_issue_count": 0,
        "annotation_issue_summary": {
            "missing_annotation": 0,
            "invalid_json": 0,
            "invalid_status": 0,
            "empty_instruction": 0,
            "other": 0,
        },
    }

    for record in records:
        descriptor_kind = classify_descriptor_storage(record.descriptor)
        stats["descriptor_paths"][descriptor_kind] = stats["descriptor_paths"].get(descriptor_kind, 0) + 1

    if preprocess_workers <= 1:
        iterator = (
            _prepare_manifest_episode(
                record,
                require_annotation,
                annotation_root,
                annotation_suffix,
                source_fps,
                target_fps,
                interpolate_labels,
            )
            for record in records
        )
    else:
        mp_context = get_context()
        pool = mp_context.Pool(preprocess_workers)
        iterator = pool.imap(
            _prepare_manifest_episode_star,
            (
                (
                    record,
                    require_annotation,
                    annotation_root,
                    annotation_suffix,
                    source_fps,
                    target_fps,
                    interpolate_labels,
                )
                for record in records
            ),
            chunksize=32,
        )

    episodes = []
    annotation_issues = []
    try:
        for episode, error_code, annotation_issue in tqdm(iterator, total=len(records), desc="Manifest episodes"):
            if annotation_issue is not None:
                annotation_issues.append(annotation_issue)
                code = str(annotation_issue.get("error_code") or "")
                if code == "missing_annotation":
                    candidates = annotation_issue.get("candidate_paths") or [annotation_issue.get("resolved_path")]
                    print(
                        "Warning: missing annotation for "
                        f"{annotation_issue.get('clip_id')}; tried: {', '.join(str(path) for path in candidates)}",
                        flush=True,
                    )
                if code in stats["annotation_issue_summary"]:
                    stats["annotation_issue_summary"][code] += 1
                else:
                    stats["annotation_issue_summary"]["other"] += 1
            if episode is None:
                stats[error_code] = stats.get(error_code, 0) + 1
                continue
            episode["episode_index"] = len(episodes)
            episodes.append(episode)
            stats["kept"] += 1
    finally:
        if preprocess_workers > 1:
            pool.close()
            pool.join()

    stats["annotation_issue_count"] = len(annotation_issues)
    return episodes, stats, annotation_issues


def _prepare_manifest_episode_star(args):
    return _prepare_manifest_episode(*args)
