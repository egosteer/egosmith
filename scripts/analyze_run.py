#!/usr/bin/env python3
"""
Analyze batch inference run results and generate report.
Usage: python scripts/analyze_run.py <run_dir>
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts_str):
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except:
        return None


def analyze_run(run_dir: Path):
    status_file = run_dir / "status.json"
    events_file = run_dir / "events.jsonl"

    if not status_file.exists():
        print(f"Error: {status_file} not found")
        return

    # Load status
    with open(status_file) as f:
        status = json.load(f)

    tasks = status.get("tasks", {})
    total_videos = len(tasks)

    # Analyze task status
    completed = []
    failed = []
    partial = []

    for video_path, task in tasks.items():
        video_name = Path(video_path).name
        stage_status = task.get("stage_status", {})

        all_completed = all(s == "completed" for s in stage_status.values())
        any_failed = any(s == "failed" for s in stage_status.values())

        if all_completed:
            completed.append(video_name)
        elif any_failed:
            failed.append((video_name, stage_status))
        else:
            partial.append((video_name, stage_status))

    # Print summary
    print("=" * 60)
    print(f"Batch Run Analysis: {run_dir.name}")
    print("=" * 60)
    print()
    print(f"Total videos: {total_videos}")
    print(f"Completed: {len(completed)} ({len(completed)/total_videos*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/total_videos*100:.1f}%)")
    print(f"Partial: {len(partial)} ({len(partial)/total_videos*100:.1f}%)")
    print()

    # Stage statistics
    stages = ["detect_track", "motion", "slam", "infiller"]
    print("Per-stage completion:")
    for stage in stages:
        stage_completed = sum(
            1 for t in tasks.values()
            if t.get("stage_status", {}).get(stage) == "completed"
        )
        print(f"  {stage:15s}: {stage_completed:3d}/{total_videos} ({stage_completed/total_videos*100:.1f}%)")
    print()

    # Failed videos detail
    if failed:
        print("Failed videos:")
        for video_name, stage_status in failed:
            failed_stages = [s for s, status in stage_status.items() if status == "failed"]
            print(f"  {video_name}")
            print(f"    Failed stages: {', '.join(failed_stages)}")
        print()

    # Partial videos detail
    if partial:
        print("Partial videos (in progress or interrupted):")
        for video_name, stage_status in partial:
            pending = [s for s, status in stage_status.items() if status == "pending"]
            running = [s for s, status in stage_status.items() if status == "running"]
            print(f"  {video_name}")
            if running:
                print(f"    Running: {', '.join(running)}")
            if pending:
                print(f"    Pending: {', '.join(pending)}")
        print()

    # Event analysis
    if events_file.exists():
        print("Event statistics:")
        event_counts = defaultdict(int)
        stage_times = defaultdict(list)
        video_times = {}

        with open(events_file) as f:
            for line in f:
                event = json.loads(line)
                event_type = event.get("event")
                event_counts[event_type] += 1

                # Track timing
                if event_type == "video_start":
                    video = event.get("video")
                    ts = parse_timestamp(event.get("time"))
                    if video and ts:
                        video_times[video] = {"start": ts}

                elif event_type == "video_completed":
                    video = event.get("video")
                    ts = parse_timestamp(event.get("time"))
                    if video and ts and video in video_times:
                        video_times[video]["end"] = ts

                elif event_type == "stage_end" and event.get("status") == "success":
                    stage = event.get("stage")
                    elapsed = event.get("elapsed_sec")
                    if stage and elapsed:
                        stage_times[stage].append(elapsed)

        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type:20s}: {count}")
        print()

        # Timing statistics
        if stage_times:
            print("Stage timing (seconds):")
            for stage in stages:
                if stage in stage_times:
                    times = stage_times[stage]
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    print(f"  {stage:15s}: avg={avg_time:6.1f}  min={min_time:6.1f}  max={max_time:6.1f}  (n={len(times)})")
            print()

        # Video processing time
        completed_times = []
        for video, times in video_times.items():
            if "start" in times and "end" in times:
                duration = (times["end"] - times["start"]).total_seconds()
                completed_times.append(duration)

        if completed_times:
            avg_video_time = sum(completed_times) / len(completed_times)
            print(f"Average video processing time: {avg_video_time:.1f} seconds ({avg_video_time/60:.1f} minutes)")
            print()

    # Recommendations
    print("Recommendations:")
    if failed:
        print("  - Check logs in logs/ directory for failed videos")
        print("  - Retry failed videos with: --run_dir", run_dir)
    if partial:
        print("  - Resume partial run with: --run_dir", run_dir)
    if not failed and not partial:
        print("  ✓ All videos completed successfully!")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze batch inference run results")
    parser.add_argument("run_dir", type=str, help="Path to batch run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Directory not found: {run_dir}")
        return 1

    analyze_run(run_dir)
    return 0


if __name__ == "__main__":
    exit(main())
