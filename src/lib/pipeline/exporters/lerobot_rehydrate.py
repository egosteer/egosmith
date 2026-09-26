"""Re-attach the video stream to a labels-only LeRobot dataset.

Input: a dataset converted with ``include_video=False`` that carries ``meta/source_frames.parquet``
(written when the converter was given the frozen clip manifest), plus the frames themselves obtained
by the user from the original data. Frames are looked up as::

    <frames_root>/<clip_id>/<frame_name>          (layout "clip_id", default)
    <frames_root>/<clip_name>/<frame_name>        (layout "clip_name")

where ``frame_name`` comes from ``source_frames.frame_names`` (the pipeline's extracted-frame names).
Output: a full dataset (video feature, mp4 files 1:1 with the data parquet files, episodes video
columns, image stats) that passes the same verification as a direct conversion.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lib.pipeline.exporters.lerobot_export import (
    HEAD_VIDEO_KEY,
    INFO_PATH,
    SOURCE_FRAMES_PATH,
    STATS_PATH,
    VIDEO_PATH,
    ConvertConfig,
    FfmpegJpegEncoder,
    _check_tools,
    _decode_jpeg_rgb,
    _ffprobe_for,
    _to_jsonable,
    aggregate_stats,
    auto_downsample_hw,
    build_features,
    chunk_file,
    image_stats_from_samples,
    jpeg_dimensions,
    probe_video_frames,
    sample_indices,
    shareable_paths,
    write_episodes_parquet,
    write_json,
)

logger = logging.getLogger(__name__)


def _episodes_rows(root: Path) -> tuple[list[dict], list[Path]]:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    tables = [pq.read_table(p) for p in paths]
    schema = tables[0].schema
    table = pa.concat_tables([t.select(schema.names).cast(schema) for t in tables])
    return table.to_pylist(), paths


REHYDRATE_README_NOTE = (
    "> **Rehydrated copy.** The video stream of this copy was re-attached locally by "
    "`scripts/build/lerobot_rehydrate_video.py` from source frames (`meta/source_frames.parquet`), so, unlike the "
    "labels-only description below, it does contain images/video. Those pixels remain under the source dataset's "
    "own terms; see `meta/release.json` (`rehydrated`)."
)


def _mark_release_rehydrated(out: Path, labels_name: str) -> None:
    """Update the copied release.json / README.md so they no longer describe the copy as labels-only."""
    release_path = out / "meta" / "release.json"
    if release_path.exists():
        release = json.loads(release_path.read_text(encoding="utf-8"))
        release["has_video"] = True
        # the copy now holds source-dataset pixels: it is a local working copy, not a redistributable release
        release["distributable"] = False
        release["rehydrated"] = {"tool": "lerobot_rehydrate_video.py", "rehydrated_from": labels_name, "video_source": "source frames re-attached locally"}
        write_json(shareable_paths(release), release_path)
    readme_path = out / "README.md"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        if REHYDRATE_README_NOTE not in text:
            readme_path.write_text(REHYDRATE_README_NOTE + "\n\n" + text, encoding="utf-8")


def rehydrate_video(labels_root: str, frames_root: str, output_dir: str, *, layout: str = "clip_id", ffmpeg: str = "ffmpeg", crf: int = 18, gop: int = 15, preset: str = "medium", ffmpeg_threads: int = 4, jpeg_ext: str = ".jpg") -> dict:
    src = Path(labels_root)
    out = Path(output_dir)
    frames = Path(frames_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    info = json.loads((src / INFO_PATH).read_text(encoding="utf-8"))
    if info.get("video_path") or HEAD_VIDEO_KEY in info["features"]:
        raise ValueError("input already has a video stream")
    sf_path = src / SOURCE_FRAMES_PATH
    if not sf_path.exists():
        raise FileNotFoundError(f"{sf_path} missing: the labels-only dataset was converted without descriptor_manifest")
    provenance = json.loads((src / "meta" / "egosmith_provenance.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    cfg = ConvertConfig(fps=fps, ffmpeg=ffmpeg, crf=crf, gop=gop, preset=preset, ffmpeg_threads=ffmpeg_threads, state_layout=provenance.get("state_layout", "egosteer74"), include_mano=any(k == "observation.mano" for k in info["features"]), include_extrinsic=any(k == "observation.camera.head_world2cam" for k in info["features"]), include_presence=any(k == "observation.hand_presence" for k in info["features"]))
    _check_tools(cfg)

    # copy everything except the meta files we regenerate
    out.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in ("_verification", "_wds_to_lerobot_work"):
            continue
        dest = out / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    rows, _ = _episodes_rows(src)
    sf_table = pq.read_table(sf_path)
    source_frames = {r["episode_index"]: r for r in sf_table.to_pylist()}
    if "frame_index_in_clip" not in sf_table.column_names or sf_table.column("frame_index_in_clip").null_count:
        logger.warning("%s has no frame_index_in_clip column or null entries (written by an older converter): frames are taken consecutively from frame_start_name, so episodes with gaps in their source frame ids cannot be checked and would be misaligned", SOURCE_FRAMES_PATH)
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["data/chunk_index"], row["data/file_index"]), []).append(row)

    height = width = None
    per_episode_image_stats: dict[int, dict] = {}
    for (chunk_index, file_index), ep_rows in sorted(groups.items()):
        video_path = out / VIDEO_PATH.format(video_key=HEAD_VIDEO_KEY, chunk_index=chunk_index, file_index=file_index)
        tmp_video = video_path.with_name(video_path.name + ".tmp.mp4")
        encoder = None
        cursor = 0.0
        try:
            for row in ep_rows:
                ep = row["episode_index"]
                sf = source_frames.get(ep)
                if sf is None:
                    raise KeyError(f"episode {ep} missing from {SOURCE_FRAMES_PATH}")
                names = list(sf["frame_names"])
                T = int(row["length"])
                if sf.get("frame_index_in_clip") is not None:
                    frame_idx = [int(i) for i in sf["frame_index_in_clip"]]
                    if len(frame_idx) != T:
                        raise ValueError(f"episode {ep}: {SOURCE_FRAMES_PATH} lists {len(frame_idx)} frame indices but the episode has {T} frames")
                    bad = [i for i in frame_idx if not 0 <= i < len(names)]
                    if bad:
                        raise ValueError(f"episode {ep}: frame indices {bad[:5]} out of range, source lists {len(names)} names")
                else:  # legacy index: consecutive frames from frame_start_name
                    if not sf["frame_start_name"] or sf["frame_start_name"] not in names:
                        raise ValueError(f"episode {ep}: frame_start_name {sf['frame_start_name']!r} is empty or not among the {len(names)} frame_names of clip {sf['clip_id']!r}")
                    start = names.index(sf["frame_start_name"])
                    if start + T > len(names):
                        raise ValueError(f"episode {ep}: needs frames {start}..{start + T} but source lists {len(names)} names")
                    frame_idx = list(range(start, start + T))
                folder = frames / (sf["clip_id"] if layout == "clip_id" else sf["clip_name"])
                samples = []
                sample_rows = set(sample_indices(T))
                for t in range(T):
                    name = names[frame_idx[t]]
                    path = folder / name
                    if not path.exists() and not name.endswith(jpeg_ext):
                        path = folder / (Path(name).stem + jpeg_ext)
                    if not path.exists():
                        raise FileNotFoundError(f"episode {ep} frame {t}: {path}")
                    data = path.read_bytes()
                    if encoder is None:
                        h, w = jpeg_dimensions(data)
                        if height is None:
                            height, width = h, w
                        encoder = FfmpegJpegEncoder(cfg, tmp_video, width, height)
                    encoder.write(data)
                    if t in sample_rows:
                        samples.append(auto_downsample_hw(_decode_jpeg_rgb(data)))
                duration = T / fps
                row[f"videos/{HEAD_VIDEO_KEY}/chunk_index"] = chunk_index
                row[f"videos/{HEAD_VIDEO_KEY}/file_index"] = file_index
                row[f"videos/{HEAD_VIDEO_KEY}/from_timestamp"] = cursor
                row[f"videos/{HEAD_VIDEO_KEY}/to_timestamp"] = cursor + duration
                cursor += duration
                stats = image_stats_from_samples(samples)
                per_episode_image_stats[ep] = stats
                for k, v in _to_jsonable(stats).items():
                    row[f"stats/{HEAD_VIDEO_KEY}/{k}"] = v
            encoder.close()
        except Exception:
            if encoder is not None:
                encoder.abort()
            raise
        video_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_video.replace(video_path)
        expected = sum(int(r["length"]) for r in ep_rows)
        probed = probe_video_frames(video_path, ffprobe=_ffprobe_for(ffmpeg), ffmpeg=ffmpeg)
        if probed is not None and probed != expected:
            raise RuntimeError(f"{video_path}: {probed} frames, expected {expected}")
        logger.info("rehydrated %s (%d episodes, %d frames)", video_path.name, len(ep_rows), expected)

    # meta/episodes: rewrite with the new columns
    shutil.rmtree(out / "meta" / "episodes")
    for row in rows:
        row.pop("meta/episodes/chunk_index", None)
        row.pop("meta/episodes/file_index", None)
    write_episodes_parquet(rows, out, cfg)

    # stats.json: add the image feature
    stats = json.loads((src / STATS_PATH).read_text(encoding="utf-8"))
    stats[HEAD_VIDEO_KEY] = _to_jsonable(aggregate_stats([{HEAD_VIDEO_KEY: per_episode_image_stats[r["episode_index"]]} for r in rows])[HEAD_VIDEO_KEY])
    write_json(stats, out / STATS_PATH)

    # info.json: video feature first (lerobot orders features as declared), then the rest
    cfg.include_video = True
    features = build_features(cfg, height, width)
    new_features = {HEAD_VIDEO_KEY: features[HEAD_VIDEO_KEY]}
    new_features.update(info["features"])
    info["features"] = new_features
    info["video_path"] = VIDEO_PATH
    write_json(info, out / INFO_PATH)
    provenance["include_video"] = True
    provenance["rehydrated_from"] = src.resolve().name  # base name only: the provenance ships with the dataset
    provenance["video_encoder"] = {"vcodec": cfg.vcodec, "crf": cfg.crf, "g": cfg.gop, "preset": cfg.preset, "pix_fmt": cfg.pix_fmt}
    write_json(shareable_paths(provenance), out / "meta" / "egosmith_provenance.json")
    _mark_release_rehydrated(out, provenance["rehydrated_from"])
    return {"output_dir": str(out), "episodes": len(rows), "frames": info["total_frames"], "frame_size": [height, width], "file_pairs": len(groups)}


__all__ = ["rehydrate_video"]
