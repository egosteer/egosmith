"""Verification of a converted LeRobot v3.0 dataset.

Three independent judges, from "does the format match" to "are the values right":

1. :func:`compare_with_reference` — structural diff against a known-good LeRobot v3 dataset
   (e.g. the EgoSteer release): info.json, feature dicts, parquet schemas and huggingface
   metadata, tasks/episodes/stats layout, video stream parameters.
2. :func:`verify_against_wds` — semantic regression against the source WebDataset: every value
   in the parquet is re-derived from the raw ``lowdim`` with an independent (naive, per-frame)
   implementation; video frames are decoded and compared to the source JPEGs (PSNR) both by
   random seeking and by streaming whole files; optional hand overlay videos for eyeballing.
3. :func:`loader_check` — the official ``lerobot`` loader, when installed: counts, windowed
   sampling across episode boundaries with padding masks, image tensors.

Every function returns a plain dict with ``ok``, ``failures``, ``warnings`` and details so the
batch driver can persist it.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
import random
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.pipeline.exporters.lerobot_export import (
    ACTION_KEY,
    EGOSTEER_ROBOT_PAD_DIM,
    EXTRINSIC_KEY,
    HEAD_VIDEO_KEY,
    INFO_PATH,
    INTRINSICS_EPISODE_KEY,
    MANO_KEY,
    PRESENCE_KEY,
    STATE_KEY,
    STATS_PATH,
    TASKS_PATH,
    INDEX_CACHE_DIRNAME,
    _pool_context,
    _worker_init,
    available_cpus,
    WORK_DIR,
    EpisodeReader,
    EpisodeRecord,
    _ffprobe_for,
    _read_member,
    collect_shards,
    decode_npy,
    index_shards,
    probe_video_frames,
)
from lib.pipeline.exporters.mano_codec import rot6d_to_rotmat as _repo_rot6d_to_rotmat
from lib.pipeline.quality.constants import LOWDIM_SIZE

logger = logging.getLogger(__name__)


# ======================================================================================
# helpers
# ======================================================================================


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _episodes_table(root: Path) -> pa.Table:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no meta/episodes parquet under {root}")
    tables = [pq.read_table(p) for p in paths]
    schema = tables[0].schema
    return pa.concat_tables([t.select(schema.names).cast(schema) for t in tables])


def _first_data_parquet(root: Path, info: dict) -> Path:
    return root / info["data_path"].format(chunk_index=0, file_index=0)


def _first_video(root: Path, info: dict, key: str) -> Path:
    return root / info["video_path"].format(video_key=key, chunk_index=0, file_index=0)


def _type_str(t: pa.DataType) -> str:
    return str(t)


def _ffprobe_stream(ffprobe: str, path: Path) -> dict:
    if shutil.which(ffprobe) is None and not Path(ffprobe).exists():
        return {}
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        streams = json.loads(result.stdout).get("streams") or []
    except json.JSONDecodeError:
        return {}
    return streams[0] if streams else {}


def _ffprobe_gop(ffprobe: str, path: Path, max_frames: int = 90) -> int | None:
    if shutil.which(ffprobe) is None and not Path(ffprobe).exists():
        return None
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=key_frame", "-of", "csv=p=0", "-read_intervals", f"%+#{max_frames}", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    flags = [line.strip().rstrip(",") for line in result.stdout.splitlines() if line.strip()]
    keys = [i for i, f in enumerate(flags) if f == "1"]
    if len(keys) >= 2:
        return keys[1] - keys[0]
    return None


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(255.0**2 / mse)


def _decode_jpeg_rgb(data: bytes) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2 failed to decode a JPEG frame")
    return img[:, :, ::-1]


def ffmpeg_decode_frame(ffmpeg: str, path: Path, timestamp: float, fps: float, width: int, height: int) -> np.ndarray:
    """Decode the single frame whose pts is ``timestamp`` (accurate seek; lands on the first frame with pts >= ss)."""
    ss = max(0.0, timestamp - 0.25 / fps)
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-ss", f"{ss:.6f}", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=False,
    )
    expected = width * height * 3
    if result.returncode != 0 or len(result.stdout) != expected:
        raise RuntimeError(f"ffmpeg seek-decode failed at {timestamp:.4f}s in {path}: rc={result.returncode}, {len(result.stdout)}/{expected} bytes; {result.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def ffmpeg_stream_frames(ffmpeg: str, path: Path, width: int, height: int, start: float | None = None, count: int | None = None):
    """Yield RGB frames of a video in order (whole file, or ``count`` frames from ``start`` seconds)."""
    cmd = [ffmpeg, "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.6f}"]
    cmd += ["-i", str(path)]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    completed = False
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise RuntimeError(f"short frame read from {path}")
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()
        completed = True
    finally:
        if not completed:
            # consumer stopped early (exception or generator close): kill quietly, never mask the cause
            proc.kill()
            proc.stdout.close()
            proc.stderr.close()
            proc.wait()
        else:
            proc.stdout.close()
            err = proc.stderr.read().decode(errors="ignore")
            proc.stderr.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg decode of {path} failed rc={rc}: {err[:300]}")


class _WdsReader:
    """Seek-based reader over the pass-1 index."""

    def __init__(self):
        self._handles: dict[str, object] = {}

    def read(self, record: EpisodeRecord, row: int, field_name: str) -> bytes:
        path = record.shard_paths[int(record.shard_idx[row])]
        handle = self._handles.get(path)
        if handle is None:
            handle = open(path, "rb")
            self._handles[path] = handle
        off = int(record.offsets[field_name][row])
        if off < 0:
            raise KeyError(f"{record.key} frame row {row} has no {field_name}")
        return _read_member(handle, off, int(record.sizes[field_name][row]))

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles = {}


# ======================================================================================
# 1. structural comparison against a reference dataset
# ======================================================================================


def _feature_signature(ft: dict) -> dict:
    sig = {"dtype": ft.get("dtype"), "rank": len(ft.get("shape") or []), "has_names": ft.get("names") is not None}
    info = ft.get("info") or {}
    if ft.get("dtype") == "video":
        sig["info_keys"] = sorted(info.keys())
        sig["codec"] = info.get("video.codec")
        sig["pix_fmt"] = info.get("video.pix_fmt")
        sig["channels"] = info.get("video.channels")
        sig["is_depth_map"] = info.get("is_depth_map", info.get("video.is_depth_map"))
    return sig


def _column_belongs_to_missing_feature(col: str, missing_features: set[str]) -> bool:
    for feature in missing_features:
        if col.startswith(f"videos/{feature}/") or col.startswith(f"stats/{feature}/"):
            return True
    return False


def compare_with_reference(ours_root: str, ref_root: str, *, ffprobe: str = "ffprobe", allowed_missing_prefixes: tuple[str, ...] = ("calibration/", "calibration.")) -> dict:
    ours = Path(ours_root)
    ref = Path(ref_root)
    failures: list[str] = []
    warnings: list[str] = []
    notes: dict = {}

    ours_info = _load_json(ours / INFO_PATH)
    ref_info = _load_json(ref / INFO_PATH)

    # --- info.json top level
    ref_keys = set(ref_info) - {"features"}
    ours_keys = set(ours_info) - {"features"}
    for key in sorted(ref_keys - ours_keys):
        failures.append(f"info.json: missing top-level key {key!r} (present in reference)")
    for key in sorted(ours_keys - ref_keys):
        warnings.append(f"info.json: extra top-level key {key!r} (not in reference)")
    for key in sorted(ref_keys & ours_keys):
        if type(ref_info[key]) is not type(ours_info[key]):
            failures.append(f"info.json: {key!r} type {type(ours_info[key]).__name__} != reference {type(ref_info[key]).__name__}")
    for key in ("codebase_version", "data_path", "video_path"):
        if key in ref_info and ref_info.get(key) != ours_info.get(key):
            failures.append(f"info.json: {key} = {ours_info.get(key)!r} != reference {ref_info.get(key)!r}")
    notes["info_values"] = {k: {"ours": ours_info.get(k), "reference": ref_info.get(k)} for k in ("fps", "chunks_size", "data_files_size_in_mb", "video_files_size_in_mb", "robot_type", "splits")}

    # --- features
    ref_feats = ref_info["features"]
    ours_feats = ours_info["features"]
    missing_features = set(ref_feats) - set(ours_feats)
    extra_features = set(ours_feats) - set(ref_feats)
    notes["features_missing_vs_reference"] = sorted(missing_features)
    notes["features_extra_vs_reference"] = sorted(extra_features)
    for key in sorted(missing_features):
        warnings.append(f"features: reference has {key!r}, ours does not (expected for chest/depth streams)")
    for key in sorted(set(ref_feats) & set(ours_feats)):
        rs, os_ = _feature_signature(ref_feats[key]), _feature_signature(ours_feats[key])
        for field_name in rs:
            if field_name == "info_keys":
                missing_info = sorted(set(rs[field_name]) - set(os_.get(field_name, [])))
                if missing_info:
                    failures.append(f"features[{key}].info: missing keys {missing_info}")
                continue
            if rs[field_name] != os_.get(field_name):
                failures.append(f"features[{key}].{field_name}: {os_.get(field_name)!r} != reference {rs[field_name]!r}")
        if ref_feats[key].get("shape") != ours_feats[key].get("shape") and ref_feats[key].get("dtype") != "video":
            failures.append(f"features[{key}].shape: {ours_feats[key].get('shape')} != reference {ref_feats[key].get('shape')}")
        elif ref_feats[key].get("dtype") == "video" and ref_feats[key].get("shape") != ours_feats[key].get("shape"):
            warnings.append(f"features[{key}].shape: {ours_feats[key].get('shape')} vs reference {ref_feats[key].get('shape')} (resolution differs)")

    # --- data parquet schema
    ours_schema = pq.read_schema(_first_data_parquet(ours, ours_info))
    ref_schema = pq.read_schema(_first_data_parquet(ref, ref_info))
    ours_cols = {f.name: f.type for f in ours_schema}
    ref_cols = {f.name: f.type for f in ref_schema}
    for name in sorted(set(ref_cols) - set(ours_cols)):
        if name in missing_features:
            continue
        failures.append(f"data parquet: missing column {name!r}")
    for name in sorted(set(ref_cols) & set(ours_cols)):
        if _type_str(ref_cols[name]) != _type_str(ours_cols[name]):
            failures.append(f"data parquet: column {name!r} type {ours_cols[name]} != reference {ref_cols[name]}")
    ours_hf = json.loads((ours_schema.metadata or {}).get(b"huggingface", b"{}") or b"{}")
    ref_hf = json.loads((ref_schema.metadata or {}).get(b"huggingface", b"{}") or b"{}")
    if ref_hf and not ours_hf:
        failures.append("data parquet: reference carries huggingface schema metadata, ours does not")
    elif ref_hf and ours_hf:
        rf = ref_hf.get("info", {}).get("features", {})
        of = ours_hf.get("info", {}).get("features", {})
        for name in sorted(set(rf) & set(of)):
            if rf[name] != of[name]:
                failures.append(f"data parquet hf metadata: {name!r} {of[name]} != reference {rf[name]}")
    notes["data_parquet_row_groups"] = {
        "ours": pq.ParquetFile(_first_data_parquet(ours, ours_info)).metadata.num_row_groups,
        "reference": pq.ParquetFile(_first_data_parquet(ref, ref_info)).metadata.num_row_groups,
    }

    # --- episodes parquet
    ours_eps = _episodes_table(ours)
    ref_eps_schema = pq.read_schema(sorted((ref / "meta" / "episodes").glob("*/*.parquet"))[0])
    ours_ep_cols = {f.name: f.type for f in ours_eps.schema}
    ref_ep_cols = {f.name: f.type for f in ref_eps_schema}
    for name in sorted(set(ref_ep_cols) - set(ours_ep_cols)):
        if _column_belongs_to_missing_feature(name, missing_features) or name.startswith(allowed_missing_prefixes):
            warnings.append(f"episodes parquet: reference column {name!r} absent (tied to a stream we do not export)")
        else:
            failures.append(f"episodes parquet: missing column {name!r}")
    for name in sorted(set(ref_ep_cols) & set(ours_ep_cols)):
        if _type_str(ref_ep_cols[name]) != _type_str(ours_ep_cols[name]):
            failures.append(f"episodes parquet: column {name!r} type {ours_ep_cols[name]} != reference {ref_ep_cols[name]}")
    notes["episodes_extra_columns"] = sorted(set(ours_ep_cols) - set(ref_ep_cols))

    # --- tasks parquet
    ours_tasks = pq.read_schema(ours / TASKS_PATH)
    ref_tasks = pq.read_schema(ref / TASKS_PATH)
    if [f.name for f in ours_tasks] != [f.name for f in ref_tasks]:
        failures.append(f"tasks parquet: columns {[f.name for f in ours_tasks]} != reference {[f.name for f in ref_tasks]}")
    for side, schema in (("ours", ours_tasks), ("reference", ref_tasks)):
        meta = json.loads((schema.metadata or {}).get(b"pandas", b"{}") or b"{}")
        notes.setdefault("tasks_index_columns", {})[side] = meta.get("index_columns")
    if notes["tasks_index_columns"]["ours"] != notes["tasks_index_columns"]["reference"]:
        failures.append(f"tasks parquet: pandas index_columns {notes['tasks_index_columns']}")

    # --- stats.json
    ours_stats = _load_json(ours / STATS_PATH)
    ref_stats = _load_json(ref / STATS_PATH)
    for key in sorted(set(ref_stats) & set(ours_stats)):
        rk, ok = sorted(ref_stats[key]), sorted(ours_stats[key])
        if rk != ok:
            failures.append(f"stats.json[{key}]: stat keys {ok} != reference {rk}")
            continue
        for stat in rk:
            rshape = np.asarray(ref_stats[key][stat]).shape
            oshape = np.asarray(ours_stats[key][stat]).shape
            if rshape != oshape and not (ref_feats[key].get("dtype") == "video"):
                failures.append(f"stats.json[{key}][{stat}]: shape {oshape} != reference {rshape}")
            elif rshape != oshape:
                failures.append(f"stats.json[{key}][{stat}]: shape {oshape} != reference {rshape}")

    # --- video streams
    ours_video = _ffprobe_stream(ffprobe, _first_video(ours, ours_info, HEAD_VIDEO_KEY)) if (ours_info.get("video_path") and HEAD_VIDEO_KEY in ours_feats) else {}
    ref_key = HEAD_VIDEO_KEY if HEAD_VIDEO_KEY in ref_feats else next((k for k, v in ref_feats.items() if v.get("dtype") == "video" and not (v.get("info") or {}).get("is_depth_map")), None)
    ref_video = _ffprobe_stream(ffprobe, _first_video(ref, ref_info, ref_key)) if ref_key else {}
    if ours_video and ref_video:
        for field_name in ("codec_name", "pix_fmt", "r_frame_rate", "avg_frame_rate"):
            if ours_video.get(field_name) != ref_video.get(field_name):
                (failures if field_name in ("codec_name", "pix_fmt") else warnings).append(
                    f"video stream {field_name}: {ours_video.get(field_name)!r} != reference {ref_video.get(field_name)!r}"
                )
        notes["video_gop"] = {
            "ours": _ffprobe_gop(ffprobe, _first_video(ours, ours_info, HEAD_VIDEO_KEY)),
            "reference": _ffprobe_gop(ffprobe, _first_video(ref, ref_info, ref_key)),
        }
        notes["video_stream"] = {"ours": ours_video, "reference": ref_video}
    elif not ours_info.get("video_path"):
        warnings.append("video stream comparison skipped: ours is a labels-only dataset without video")
    else:
        warnings.append("video stream comparison skipped (ffprobe unavailable or video missing)")

    return {"ok": not failures, "failures": failures, "warnings": warnings, "notes": notes, "ours": str(ours), "reference": str(ref)}


# ======================================================================================
# 2. semantic regression against the source WebDataset
# ======================================================================================


def _naive_rot6d_to_cam(rot6d_world: np.ndarray, R_w2c: np.ndarray) -> np.ndarray:
    """Independent re-derivation: repo Gram-Schmidt -> rotate -> first two columns, column-major."""
    Rw = _repo_rot6d_to_rotmat(rot6d_world.reshape(1, 6))[0]
    Rc = R_w2c @ Rw
    return np.array([Rc[0, 0], Rc[1, 0], Rc[2, 0], Rc[0, 1], Rc[1, 1], Rc[2, 1]], dtype=np.float32)


def naive_expected_hand_block(row48: np.ndarray, w2c: np.ndarray | None) -> np.ndarray:
    """Per-frame, loop-based oracle for the 48-d EgoSteer hand block (left wrist 9, right wrist 9, tips 30)."""
    def pt(p):
        if w2c is None:
            return np.asarray(p, dtype=np.float32)
        q = w2c @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        return q[:3].astype(np.float32)

    def rot(r):
        if w2c is None:
            return np.asarray(r, dtype=np.float32)
        return _naive_rot6d_to_cam(np.asarray(r, dtype=np.float32), w2c[:3, :3])

    out = []
    out.extend(pt(row48[0:3]))
    out.extend(rot(row48[6:12]))
    out.extend(pt(row48[3:6]))
    out.extend(rot(row48[12:18]))
    for k in range(5):
        out.extend(pt(row48[18 + 3 * k : 21 + 3 * k]))
    for k in range(5):
        out.extend(pt(row48[33 + 3 * k : 36 + 3 * k]))
    return np.array(out, dtype=np.float32)


def _expected_task(meta: dict, task_source: str, fallback_key: str) -> str:
    instructions = [str(x) for x in (meta.get("instruction") or []) if str(x).strip()]
    language = str(meta.get("language") or "").strip()
    dataset_name = str(meta.get("dataset_name") or "")
    chain = {
        "language": [language, instructions[0] if instructions else "", dataset_name],
        "first_instruction": [instructions[0] if instructions else "", language, dataset_name],
        "dataset_name": [dataset_name, language],
        "clip_id": [str(meta.get("clip_id") or fallback_key)],
    }[task_source]
    return next((c for c in chain if c), "")


@dataclass
class _EpisodeLocator:
    episodes: dict
    fps: float
    from_index: list[int]

    def locate(self, global_index: int) -> tuple[int, int]:
        """-> (episode position, local frame)"""
        pos = bisect.bisect_right(self.from_index, global_index) - 1
        return pos, global_index - self.from_index[pos]


def _episode_ctx(root: Path, info: dict, episodes: dict, ep_pos: int) -> dict:
    keys = ("episode_index", "length", "data/chunk_index", "data/file_index", "dataset_from_index", "tasks", "instructions", "split", INTRINSICS_EPISODE_KEY)
    ctx = {k: episodes[k][ep_pos] for k in keys if k in episodes}
    if f"videos/{HEAD_VIDEO_KEY}/chunk_index" in episodes:
        for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            ctx[f"videos/{HEAD_VIDEO_KEY}/{k}"] = episodes[f"videos/{HEAD_VIDEO_KEY}/{k}"][ep_pos]
    ctx["ep_pos"] = ep_pos
    ctx["data_path"] = str(root / info["data_path"].format(chunk_index=ctx["data/chunk_index"], file_index=ctx["data/file_index"]))
    return ctx


def _read_episode_rows(ctx: dict) -> dict:
    return pq.read_table(ctx["data_path"], filters=[("episode_index", "==", int(ctx["episode_index"]))]).to_pydict()


def _table_to_arrays(table: pa.Table) -> dict[str, np.ndarray]:
    """Whole data parquet -> numpy per column (2-D for list columns), one conversion per file."""
    out: dict[str, np.ndarray] = {}
    for name in table.column_names:
        col = table.column(name).combine_chunks()
        if pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
            flat = col.flatten().to_numpy(zero_copy_only=False)
            out[name] = flat.reshape(len(col), -1) if len(col) else flat.reshape(0, 0)
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
    return out


def _numeric_check_file(payload: dict) -> list[tuple[int, list[str], float]]:
    """Read one data parquet once and check every selected episode inside it (worker process)."""
    arrays = _table_to_arrays(pq.read_table(payload["file"]))
    ep_col = arrays["episode_index"]
    handles: dict[int, object] = {}
    results = []
    try:
        for ctx, rec in payload["episodes"]:
            rows_idx = np.nonzero(ep_col == int(ctx["episode_index"]))[0]
            rows = {name: arr[rows_idx] for name, arr in arrays.items()}
            results.append(_numeric_check_episode({"ctx": ctx, "record": rec, "cfg": payload["cfg"], "rows": rows, "handles": handles}))
    finally:
        for h in handles.values():
            h.close()
    return results


def _numeric_check_episode(payload: dict) -> tuple[int, list[str], float]:
    """Re-derive every value of one episode from the raw lowdim (runs in a worker process)."""
    ctx, rec, cfg = payload["ctx"], payload["record"], payload["cfg"]
    ep_pos = ctx["ep_pos"]
    fps = cfg["fps"]
    failures: list[str] = []
    max_abs_err = 0.0
    handles = payload.get("handles")
    own_handles = handles is None
    if own_handles:
        handles = {}
    try:
        rows = payload.get("rows")
        if rows is None:
            rows = _read_episode_rows(ctx)
        reader = EpisodeReader(rec, handles)
        T = rec.length
        if len(rows["index"]) != T:
            return ep_pos, [f"episode {ep_pos}: parquet has {len(rows['index'])} rows, expected {T}"], 0.0
        state = np.asarray(rows[STATE_KEY], dtype=np.float32)
        action = np.asarray(rows[ACTION_KEY], dtype=np.float32)
        for t in range(T):
            lowdim = decode_npy(reader.read(t, "lowdim_bytes")).reshape(-1)[:LOWDIM_SIZE]
            meta = json.loads(reader.read(t, "meta_bytes").decode("utf-8"))
            w2c = lowdim[96:112].reshape(4, 4).astype(np.float64)
            frame_w2c = w2c if cfg["hand_frame"] == "camera" else None
            exp_state = naive_expected_hand_block(lowdim[0:48], frame_w2c)
            exp_action = naive_expected_hand_block(lowdim[48:96], frame_w2c)
            if cfg["state_layout"] == "egosteer74":
                pad = np.full((EGOSTEER_ROBOT_PAD_DIM,), cfg["pad_value"], dtype=np.float32)
                exp_state = np.concatenate([pad, exp_state])
                exp_action = np.concatenate([pad, exp_action])
            err = max(float(np.abs(state[t] - exp_state).max()), float(np.abs(action[t] - exp_action).max()))
            max_abs_err = max(max_abs_err, err)
            if err > cfg["atol"]:
                failures.append(f"episode {ep_pos} frame {t}: state/action mismatch, max abs err {err:.3e}")
                break
            if cfg["hand_frame"] == "camera" and cfg["state_layout"] == "egosteer74":
                c2w = np.linalg.inv(w2c)
                p = c2w @ np.array([*state[t, 26:29], 1.0])
                if np.abs(p[:3] - lowdim[0:3]).max() > 1e-3:
                    failures.append(f"episode {ep_pos} frame {t}: inverse transform of left wrist does not recover world lowdim")
                    break
            if cfg["has_extrinsic"] and np.abs(np.asarray(rows[EXTRINSIC_KEY][t]) - lowdim[96:112]).max() > 1e-6:
                failures.append(f"episode {ep_pos} frame {t}: extrinsic column != lowdim[96:112]")
                break
            if cfg["has_presence"] and int(rows[PRESENCE_KEY][t]) != int(meta.get("presence", 0)):
                failures.append(f"episode {ep_pos} frame {t}: presence {rows[PRESENCE_KEY][t]} != meta {meta.get('presence')}")
                break
            if cfg["has_mano"]:
                mano = decode_npy(reader.read(t, "mano_bytes")).reshape(-1)
                if np.abs(np.asarray(rows[MANO_KEY][t], dtype=np.float32) - mano).max() > 1e-6:
                    failures.append(f"episode {ep_pos} frame {t}: mano column != mano.npy")
                    break
            if abs(float(rows["timestamp"][t]) - t / fps) > 1e-4 or int(rows["frame_index"][t]) != t:
                failures.append(f"episode {ep_pos} frame {t}: timestamp/frame_index wrong ({rows['timestamp'][t]}, {rows['frame_index'][t]})")
                break
            if int(rows["index"][t]) != ctx["dataset_from_index"] + t or int(rows["episode_index"][t]) != ctx["episode_index"]:
                failures.append(f"episode {ep_pos} frame {t}: global index / episode_index wrong")
                break
            if t == 0:
                exp_task = cfg.get("task_name") or _expected_task(meta, cfg["task_source"], rec.key)
                got_task = cfg["task_names"].get(int(rows["task_index"][t]))
                if got_task != exp_task:
                    failures.append(f"episode {ep_pos}: task {got_task!r} != expected {exp_task!r}")
                if ctx.get("tasks") != [exp_task]:
                    failures.append(f"episode {ep_pos}: episodes.tasks {ctx.get('tasks')} != [{exp_task!r}]")
                if "instructions" in ctx and list(ctx["instructions"]) != [str(x) for x in (meta.get("instruction") or []) if str(x).strip()]:
                    failures.append(f"episode {ep_pos}: instructions differ from meta.instruction")
                if INTRINSICS_EPISODE_KEY in ctx:
                    fx, fy, cx, cy = (float(v) for v in lowdim[112:116])
                    if not np.allclose(ctx[INTRINSICS_EPISODE_KEY], [fx, 0, cx, 0, fy, cy, 0, 0, 1], atol=1e-4):
                        failures.append(f"episode {ep_pos}: intrinsics row differs from lowdim[112:116]")
                if "split" in ctx and ctx["split"] != str(meta.get("split") or "train"):
                    failures.append(f"episode {ep_pos}: split {ctx['split']!r} != meta {meta.get('split')!r}")
                if failures:
                    break
    finally:
        if own_handles:
            for h in handles.values():
                h.close()
    return ep_pos, failures, max_abs_err


def _random_frame_check(payload: dict) -> dict:
    """Seek-decode one frame and compare with the source JPEG (runs in a thread; subprocess-bound)."""
    ctx, rec, cfg, gi, local = payload["ctx"], payload["record"], payload["cfg"], payload["gi"], payload["local"]
    vpath = Path(cfg["root"]) / cfg["video_path"].format(video_key=HEAD_VIDEO_KEY, chunk_index=ctx[f"videos/{HEAD_VIDEO_KEY}/chunk_index"], file_index=ctx[f"videos/{HEAD_VIDEO_KEY}/file_index"])
    ts = float(ctx[f"videos/{HEAD_VIDEO_KEY}/from_timestamp"]) + local / cfg["fps"]
    reader = _WdsReader()
    try:
        decoded = ffmpeg_decode_frame(cfg["ffmpeg"], vpath, ts, cfg["fps"], cfg["width"], cfg["height"])
        value = psnr(decoded, _decode_jpeg_rgb(reader.read(rec, local, "image_bytes")))
        neighbours = {}
        if value < cfg["min_psnr"]:
            for d in (-1, 1):
                if 0 <= local + d < rec.length:
                    neighbours[d] = psnr(decoded, _decode_jpeg_rgb(reader.read(rec, local + d, "image_bytes")))
        return {"gi": gi, "ep_pos": ctx["ep_pos"], "local": local, "psnr": value, "neighbours": neighbours}
    except RuntimeError as error:
        return {"gi": gi, "ep_pos": ctx["ep_pos"], "local": local, "error": str(error)}
    finally:
        reader.close()


def _full_file_check(payload: dict) -> dict:
    """Stream-decode one mp4 and compare every k-th frame with the source JPEGs (runs in a worker process)."""
    cfg = payload["cfg"]
    chunk_index, file_index = payload["chunk_index"], payload["file_index"]
    seq = payload["seq"]  # list of (ep_pos, local, record)
    vpath = Path(cfg["root"]) / cfg["video_path"].format(video_key=HEAD_VIDEO_KEY, chunk_index=chunk_index, file_index=file_index)
    expected = payload["expected_frames"]
    failures: list[str] = []
    probed = probe_video_frames(vpath, ffprobe=_ffprobe_for(cfg["ffmpeg"]), ffmpeg=cfg["ffmpeg"])
    if probed is not None and probed != expected:
        failures.append(f"{vpath}: container reports {probed} frames, episodes say {expected}")
    handles: dict[int, object] = {}
    readers: dict[int, EpisodeReader] = {}
    n_decoded = 0
    fp: list[float] = []
    worst = None
    try:
        for k, frame in enumerate(ffmpeg_stream_frames(cfg["ffmpeg"], vpath, cfg["width"], cfg["height"])):
            n_decoded = k + 1
            if k >= len(seq) or k % cfg["full_stride"]:
                continue
            ep_pos, t, rec = seq[k]
            reader = readers.get(ep_pos)
            if reader is None:
                readers.clear()  # episodes arrive in file order; keep only the current block in memory
                reader = readers[ep_pos] = EpisodeReader(rec, handles, fields=("image_bytes",))
            value = psnr(frame, _decode_jpeg_rgb(reader.read(t, "image_bytes")))
            fp.append(value)
            if worst is None or value < worst[0]:
                worst = (value, ep_pos, t)
            if value < cfg["min_psnr"]:
                failures.append(f"{vpath} frame {k} (episode {ep_pos}, local {t}): PSNR {value:.1f} dB < {cfg['min_psnr']}")
    finally:
        for h in handles.values():
            h.close()
    if n_decoded != expected:
        failures.append(f"{vpath}: decoded {n_decoded} frames, episodes say {expected}")
    return {"chunk_index": chunk_index, "file_index": file_index, "frames": n_decoded, "compared": len(fp), "psnr_min": min(fp) if fp else None, "psnr_median": float(np.median(fp)) if fp else None, "worst": worst, "failures": failures}


def verify_against_wds(
    lerobot_root: str,
    wds_inputs: list[str],
    *,
    ffmpeg: str = "ffmpeg",
    random_frames: int = 200,
    full_files: int = 1,
    full_stride: int = 1,
    numeric_episodes: int | None = None,
    overlay_episodes: int = 2,
    overlay_dir: str | None = None,
    min_psnr: float = 30.0,
    atol: float = 2e-4,
    seed: int = 0,
    workers: int = 0,
) -> dict:
    """``workers``: process/thread pool size for the per-episode, per-frame and per-file checks (0 = cpu count)."""
    root = Path(lerobot_root)
    rng = random.Random(seed)
    failures: list[str] = []
    warnings: list[str] = []
    notes: dict = {}
    workers = workers if workers > 0 else available_cpus()

    info = _load_json(root / INFO_PATH)
    provenance_path = root / "meta" / "egosmith_provenance.json"
    provenance = _load_json(provenance_path) if provenance_path.exists() else {}
    fps = float(info["fps"])
    has_video = HEAD_VIDEO_KEY in info["features"] and bool(info.get("video_path"))
    if has_video:
        height, width = info["features"][HEAD_VIDEO_KEY]["shape"][:2]
    else:
        height = width = 0
        random_frames = 0
        full_files = 0
        overlay_episodes = 0
        warnings.append("dataset has no video stream (labels-only): frame alignment and overlay checks skipped")

    episodes = _episodes_table(root).to_pydict()
    n_eps = len(episodes["episode_index"])
    tasks_tbl = pq.read_table(root / TASKS_PATH).to_pydict()
    task_names = {int(i): str(t) for t, i in zip(tasks_tbl["task"], tasks_tbl["task_index"])}
    locator = _EpisodeLocator(episodes, fps, list(episodes["dataset_from_index"]))
    cfg = {
        "root": str(root), "fps": fps, "ffmpeg": ffmpeg, "width": width, "height": height, "min_psnr": min_psnr, "atol": atol,
        "full_stride": max(1, int(full_stride)), "video_path": info.get("video_path"),
        "hand_frame": provenance.get("hand_frame", "camera"), "state_layout": provenance.get("state_layout", "egosteer74"),
        "pad_value": float(provenance.get("pad_value", 0.0)), "task_source": provenance.get("task_source", "language"), "task_name": provenance.get("task_name"),
        "task_names": task_names, "has_mano": MANO_KEY in info["features"], "has_presence": PRESENCE_KEY in info["features"],
        "has_extrinsic": EXTRINSIC_KEY in info["features"],
    }

    # --- WDS index and clip mapping
    shard_paths = collect_shards(wds_inputs)
    records = index_shards(shard_paths, workers=workers, cache_dir=root / WORK_DIR / INDEX_CACHE_DIRNAME)
    by_clip: dict[str, EpisodeRecord] = {}
    by_episode: dict[tuple[str, int], EpisodeRecord] = {}  # repeated clips (``<clip>__rep<k>`` keys) share a clip_id
    for r in records:
        by_clip[str(r.meta.get("clip_id") or r.key)] = r
        by_clip.setdefault(r.key, r)
        by_episode[(str(r.meta.get("clip_id") or r.key), int(r.source_episode_index))] = r
    if "clip_id" not in episodes:
        raise ValueError("episodes metadata has no clip_id column; cannot map back to the WebDataset")
    matched: list[EpisodeRecord | None] = []
    for i in range(n_eps):
        rec = None
        if "source_episode_index" in episodes and episodes["source_episode_index"][i] is not None:
            rec = by_episode.get((str(episodes["clip_id"][i]), int(episodes["source_episode_index"][i])))
        if rec is None:
            rec = by_clip.get(str(episodes["clip_id"][i]))
        if rec is None:
            failures.append(f"episode {i}: clip_id {episodes['clip_id'][i]!r} not found in the WebDataset")
        elif rec.length != episodes["length"][i]:
            failures.append(f"episode {i}: length {episodes['length'][i]} != WDS frames {rec.length}")
        matched.append(rec)
    dropped = (provenance.get("dropped_episodes") or {})
    dropped_keys = set(dropped.get("keys") or [])
    kept = [r for r in records if r.key not in dropped_keys]
    if dropped_keys:
        notes["dropped_episodes"] = {"reason": dropped.get("reason"), "count": len(dropped_keys)}
        warnings.append(f"{len(dropped_keys)} WDS episodes were dropped at conversion ({dropped.get('reason')}); excluded from the counts")
    wds_frames = sum(r.length for r in kept)
    if wds_frames != info["total_frames"]:
        failures.append(f"WDS has {wds_frames} frames (after drops), dataset has {info['total_frames']}")
    if len(kept) != n_eps:
        failures.append(f"WDS has {len(kept)} episodes (after drops), dataset has {n_eps}")

    # --- numeric regression (process pool over episodes)
    candidates = [i for i in range(n_eps) if matched[i] is not None and matched[i].length == episodes["length"][i]]
    if numeric_episodes is not None and numeric_episodes < len(candidates):
        chosen = sorted(rng.sample(candidates, numeric_episodes))
    else:
        chosen = candidates
    by_file: dict[str, list] = {}
    for i in chosen:
        ctx = _episode_ctx(root, info, episodes, i)
        by_file.setdefault(ctx["data_path"], []).append((ctx, matched[i]))
    payloads = [{"file": path, "episodes": eps, "cfg": cfg} for path, eps in by_file.items()]
    numeric_checked = 0
    max_abs_err = 0.0
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads)), mp_context=_pool_context(), initializer=_worker_init) as pool:
            results = [r for batch in pool.map(_numeric_check_file, payloads, chunksize=1) for r in batch]
    else:
        results = [r for p in payloads for r in _numeric_check_file(p)]
    for _ep_pos, ep_failures, err in results:
        max_abs_err = max(max_abs_err, err)
        if ep_failures:
            failures.extend(ep_failures)
        else:
            numeric_checked += 1
    notes["numeric_episodes_checked"] = numeric_checked
    notes["numeric_max_abs_err"] = max_abs_err
    notes["workers"] = workers

    # --- splits contiguity from meta
    if "split" in episodes:
        seen = []
        for s in episodes["split"]:
            if not seen or seen[-1] != s:
                seen.append(s)
        if len(seen) != len(set(seen)):
            failures.append(f"splits are not contiguous in episode order: {seen}")
        for name, rng_str in info.get("splits", {}).items():
            a, b = (int(x) for x in rng_str.split(":"))
            if any(episodes["split"][i] != name for i in range(a, b)):
                failures.append(f"info.splits[{name}]={rng_str} does not match episodes.split")

    # --- random frame alignment (thread pool; each check is an ffmpeg subprocess)
    total_frames = int(info["total_frames"])
    if random_frames > 0 and total_frames > 0:
        picks = sorted(rng.sample(range(total_frames), min(random_frames, total_frames)))
        jobs = []
        for gi in picks:
            ep_pos, local = locator.locate(gi)
            if matched[ep_pos] is None:
                continue
            jobs.append({"ctx": _episode_ctx(root, info, episodes, ep_pos), "record": matched[ep_pos], "cfg": cfg, "gi": gi, "local": local})
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(jobs)))) as pool:
            results = list(pool.map(_random_frame_check, jobs))
        psnrs = []
        worst = None
        for r in results:
            if "error" in r:
                failures.append(r["error"])
                continue
            psnrs.append(r["psnr"])
            if worst is None or r["psnr"] < worst[0]:
                worst = (r["psnr"], r["gi"], r["ep_pos"], r["local"])
            if r["psnr"] < min_psnr:
                failures.append(f"frame {r['gi']} (episode {r['ep_pos']}, local {r['local']}): PSNR {r['psnr']:.1f} dB < {min_psnr}; neighbours {r['neighbours']}")
        if psnrs:
            notes["random_frames"] = {"count": len(psnrs), "psnr_min": min(psnrs), "psnr_median": float(np.median(psnrs)), "psnr_mean": float(np.mean(psnrs)), "worst": worst}

    # --- whole-file streaming decode (process pool over files)
    file_keys = sorted({(episodes["data/chunk_index"][i], episodes["data/file_index"][i]) for i in range(n_eps)})
    if full_files > 0 and file_keys:
        pick_files = [file_keys[0]]
        if len(file_keys) > 1:
            pick_files.append(file_keys[-1])
        others = [k for k in file_keys if k not in pick_files]
        rng.shuffle(others)
        pick_files.extend(others[: max(0, full_files - len(pick_files))])
        pick_files = pick_files[:full_files]
        jobs = []
        for chunk_index, file_index in pick_files:
            ep_positions = [i for i in range(n_eps) if (episodes["data/chunk_index"][i], episodes["data/file_index"][i]) == (chunk_index, file_index)]
            seq = [(i, t, matched[i]) for i in ep_positions if matched[i] is not None for t in range(episodes["length"][i])]
            jobs.append({"cfg": cfg, "chunk_index": chunk_index, "file_index": file_index, "seq": seq, "expected_frames": sum(episodes["length"][i] for i in ep_positions)})
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=_pool_context(), initializer=_worker_init) as pool:
                results = list(pool.map(_full_file_check, jobs))
        else:
            results = [_full_file_check(j) for j in jobs]
        full_report = []
        for r in results:
            failures.extend(r.pop("failures"))
            full_report.append(r)
        notes["full_files"] = full_report

    # --- overlay videos (thread pool; ffmpeg-bound)
    if overlay_episodes > 0 and overlay_dir:
        out_dir = Path(overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        picks = sorted(rng.sample(candidates, min(overlay_episodes, len(candidates))))

        def _one(ep_pos):
            try:
                return str(render_overlay(root, info, episodes, ep_pos, out_dir, ffmpeg=ffmpeg, hand_frame=cfg["hand_frame"], state_layout=cfg["state_layout"]))
            except Exception as error:  # overlay is diagnostic; never masks other checks
                warnings.append(f"overlay for episode {ep_pos} failed: {error!r}")
                return None

        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(picks)))) as pool:
            made = [m for m in pool.map(_one, picks) if m]
        notes["overlays"] = made

    return {"ok": not failures, "failures": failures, "warnings": warnings, "notes": notes, "lerobot_root": str(root), "wds": shard_paths}


def render_overlay(root: Path, info: dict, episodes: dict, ep_pos: int, out_dir: Path, *, ffmpeg: str, hand_frame: str, state_layout: str) -> Path:
    """Project wrists + fingertips through the head intrinsics onto the decoded video frames."""
    import cv2

    fps = float(info["fps"])
    height, width = info["features"][HEAD_VIDEO_KEY]["shape"][:2]
    rows = pq.read_table(
        root / info["data_path"].format(chunk_index=episodes["data/chunk_index"][ep_pos], file_index=episodes["data/file_index"][ep_pos]),
        filters=[("episode_index", "==", int(episodes["episode_index"][ep_pos]))],
    ).to_pydict()
    T = len(rows["index"])
    state = np.asarray(rows[STATE_KEY], dtype=np.float32)
    off = EGOSTEER_ROBOT_PAD_DIM if state_layout == "egosteer74" else 0
    K = np.asarray(episodes[INTRINSICS_EPISODE_KEY][ep_pos], dtype=np.float64).reshape(3, 3)
    vpath = root / info["video_path"].format(
        video_key=HEAD_VIDEO_KEY,
        chunk_index=episodes[f"videos/{HEAD_VIDEO_KEY}/chunk_index"][ep_pos],
        file_index=episodes[f"videos/{HEAD_VIDEO_KEY}/file_index"][ep_pos],
    )
    start = float(episodes[f"videos/{HEAD_VIDEO_KEY}/from_timestamp"][ep_pos])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_ep{int(episodes['episode_index'][ep_pos]):06d}.mp4"
    writer = subprocess.Popen(
        [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(int(fps)), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    colors = {"left": (255, 80, 80), "right": (80, 160, 255)}
    try:
        n_written = 0
        for t, frame in enumerate(ffmpeg_stream_frames(ffmpeg, vpath, width, height, start=start - 0.25 / fps, count=T)):
            img = frame  # writable copy from the decoder
            block = state[t, off:]
            w2c = None
            if hand_frame == "world" and EXTRINSIC_KEY in rows:
                w2c = np.asarray(rows[EXTRINSIC_KEY][t], dtype=np.float64).reshape(4, 4)
            pts = {
                "left": [block[0:3]] + [block[18 + 3 * k : 21 + 3 * k] for k in range(5)],
                "right": [block[9:12]] + [block[33 + 3 * k : 36 + 3 * k] for k in range(5)],
            }
            for side, plist in pts.items():
                for j, p in enumerate(plist):
                    p = np.asarray(p, dtype=np.float64)
                    if w2c is not None:
                        p = (w2c @ np.array([*p, 1.0]))[:3]
                    if p[2] <= 1e-6:
                        continue
                    uv = K @ (p / p[2])
                    u, v = int(round(uv[0])), int(round(uv[1]))
                    if 0 <= u < width and 0 <= v < height:
                        cv2.circle(img, (u, v), 6 if j == 0 else 3, colors[side], -1 if j == 0 else 2)
            cv2.putText(img, f"ep{int(episodes['episode_index'][ep_pos])} f{t} {hand_frame}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            writer.stdin.write(img.tobytes())
            n_written += 1
    finally:
        writer.stdin.close()
        err = writer.stderr.read().decode(errors="ignore")
        rc = writer.wait()
        if rc != 0:
            raise RuntimeError(f"overlay encode failed rc={rc}: {err[:300]}")
    if n_written != T:
        raise RuntimeError(f"overlay for episode {ep_pos}: decoded {n_written} frames, expected {T}")
    return out_path


# ======================================================================================
# 3. official lerobot loader
# ======================================================================================


def loader_check(root: str, *, num_items: int = 64, window_state: int = 16, window_action: int = 32, seed: int = 0, batch_size: int = 8) -> dict:
    """Load through ``lerobot.datasets.lerobot_dataset.LeRobotDataset`` (0.6.x API). Skipped when lerobot is missing."""
    root_path = Path(root)
    failures: list[str] = []
    warnings: list[str] = []
    notes: dict = {}
    try:
        import lerobot  # type: ignore  # noqa: F401
    except ImportError:
        return {"ok": True, "skipped": "lerobot not installed", "failures": [], "warnings": ["loader check skipped: lerobot not installed"], "notes": {}}
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception as error:  # lerobot is present but its dataset stack cannot import (missing av/datasets/torchcodec, ...)
        return {
            "ok": False,
            "failures": [f"lerobot is installed but `lerobot.datasets.lerobot_dataset` failed to import: {error!r}. Fix the environment (e.g. pip install av datasets) and rerun."],
            "warnings": [],
            "notes": {"traceback": traceback.format_exc()},
        }

    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
    except ImportError:
        return {"ok": True, "skipped": "torch not installed", "failures": [], "warnings": ["loader check skipped: torch not installed"], "notes": {}}

    info = _load_json(root_path / INFO_PATH)
    fps = int(info["fps"])
    repo_id = f"local/{root_path.name}"
    rng = random.Random(seed)

    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(root_path))
    except Exception as error:
        return {"ok": False, "failures": [f"LeRobotDataset() failed: {error!r}"], "warnings": [], "notes": {}}
    try:
        lerobot_version = __import__("lerobot").__version__
    except Exception:
        lerobot_version = None
    notes["lerobot_version"] = lerobot_version
    notes["num_frames"] = int(ds.num_frames)
    notes["num_episodes"] = int(ds.num_episodes)
    if ds.num_frames != info["total_frames"]:
        failures.append(f"loader num_frames {ds.num_frames} != info.total_frames {info['total_frames']}")
    if ds.num_episodes != info["total_episodes"]:
        failures.append(f"loader num_episodes {ds.num_episodes} != info.total_episodes {info['total_episodes']}")
    meta = ds.meta
    try:
        notes["num_tasks"] = int(meta.total_tasks)
        if meta.total_tasks != info["total_tasks"]:
            failures.append(f"loader total_tasks {meta.total_tasks} != info.total_tasks {info['total_tasks']}")
    except Exception as error:
        warnings.append(f"could not read meta.total_tasks: {error!r}")
    try:
        eps = meta.episodes
        from_idx = list(eps["dataset_from_index"])
        to_idx = list(eps["dataset_to_index"])
        if from_idx[0] != 0 or any(to_idx[i] != from_idx[i + 1] for i in range(len(from_idx) - 1)) or to_idx[-1] != ds.num_frames:
            failures.append("meta.episodes dataset_from/to_index not contiguous")
    except Exception as error:
        warnings.append(f"could not verify meta.episodes contiguity: {error!r}")

    # single items (video decoding, shapes, ranges)
    picks = sorted(rng.sample(range(ds.num_frames), min(num_items, ds.num_frames)))
    shapes: dict[str, list[int]] = {}
    for i in picks:
        try:
            item = ds[i]
        except Exception as error:
            failures.append(f"ds[{i}] failed: {error!r}")
            break
        for key, value in item.items():
            if hasattr(value, "shape"):
                shapes.setdefault(key, list(value.shape))
        if HEAD_VIDEO_KEY in info["features"]:
            img = item.get(HEAD_VIDEO_KEY)
            if img is None:
                failures.append(f"ds[{i}] has no {HEAD_VIDEO_KEY}")
                break
            if tuple(img.shape) != (3, *info["features"][HEAD_VIDEO_KEY]["shape"][:2]):
                failures.append(f"ds[{i}] image shape {tuple(img.shape)} != (3,H,W)")
                break
            if float(img.min()) < 0.0 or float(img.max()) > 1.0:
                failures.append(f"ds[{i}] image values outside [0,1]")
                break
        if not np.isfinite(np.asarray(item[STATE_KEY])).all():
            failures.append(f"ds[{i}] state has non-finite values")
            break
    notes["item_shapes"] = shapes

    # windowed sampling across episode boundaries with padding masks
    delta = {
        STATE_KEY: [-(k) / fps for k in range(window_state - 1, -1, -1)],
        ACTION_KEY: [k / fps for k in range(window_action)],
    }
    try:
        dsw = LeRobotDataset(repo_id=repo_id, root=str(root_path), delta_timestamps=delta)
        eps = meta.episodes
        boundary = []
        n = min(4, ds.num_episodes)
        for e in range(n):
            boundary.append(int(eps[e]["dataset_from_index"]))
            boundary.append(int(eps[e]["dataset_to_index"]) - 1)
        for i in boundary:
            item = dsw[i]
            s = item[STATE_KEY]
            a = item[ACTION_KEY]
            if tuple(s.shape)[0] != window_state or tuple(a.shape)[0] != window_action:
                failures.append(f"windowed ds[{i}]: state {tuple(s.shape)} / action {tuple(a.shape)} window lengths wrong")
                break
            spad = item.get(f"{STATE_KEY}_is_pad")
            apad = item.get(f"{ACTION_KEY}_is_pad")
            if spad is None or apad is None:
                failures.append(f"windowed ds[{i}]: *_is_pad masks missing")
                break
            at_start = any(int(eps[e]["dataset_from_index"]) == i for e in range(n))
            at_end = any(int(eps[e]["dataset_to_index"]) - 1 == i for e in range(n))
            if at_start and not bool(spad[0]) and window_state > 1:
                failures.append(f"windowed ds[{i}] (episode start): state history should be padded")
                break
            if at_end and not bool(apad[-1]) and window_action > 1:
                failures.append(f"windowed ds[{i}] (episode end): action future should be padded")
                break
        notes["window_check"] = {"indices": boundary, "state_window": window_state, "action_window": window_action}
    except Exception as error:
        failures.append(f"windowed sampling failed: {error!r}")

    # DataLoader pass over a random subset
    try:
        subset = torch_subset(ds, sorted(rng.sample(range(ds.num_frames), min(num_items, ds.num_frames))))
        loader = DataLoader(subset, batch_size=batch_size, num_workers=0, shuffle=False)
        n_batches = 0
        for batch in loader:
            n_batches += 1
            if HEAD_VIDEO_KEY in info["features"] and tuple(batch[HEAD_VIDEO_KEY].shape)[1:] != (3, *info["features"][HEAD_VIDEO_KEY]["shape"][:2]):
                failures.append(f"DataLoader batch image shape {tuple(batch[HEAD_VIDEO_KEY].shape)}")
                break
        notes["dataloader_batches"] = n_batches
    except Exception as error:
        failures.append(f"DataLoader pass failed: {error!r}")

    return {"ok": not failures, "failures": failures, "warnings": warnings, "notes": notes}


def torch_subset(ds, indices: list[int]):
    from torch.utils.data import Subset

    return Subset(ds, indices)


__all__ = [
    "compare_with_reference",
    "ffmpeg_decode_frame",
    "ffmpeg_stream_frames",
    "loader_check",
    "naive_expected_hand_block",
    "psnr",
    "render_overlay",
    "verify_against_wds",
]
