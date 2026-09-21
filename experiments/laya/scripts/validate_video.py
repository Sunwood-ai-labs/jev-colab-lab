"""Validate a JevDash MP4 against its sanitized episode JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--episode-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    args = parser.parse_args()

    episode = json.loads(args.episode_json.read_text(encoding="utf-8"))
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise RuntimeError("ffprobe and ffmpeg are required for video validation")

    probe = run_command([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(args.video)])
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr[-1200:]}")
    probe_json = json.loads(probe.stdout)
    stream = next(item for item in probe_json["streams"] if item.get("codec_type") == "video")

    decode = run_command([ffmpeg, "-v", "error", "-i", str(args.video), "-map", "0:v:0", "-f", "null", "-"])
    expected_frames = int(episode["episode"]["video_frames"])
    decoded_frames = int(stream.get("nb_read_frames", "0"))
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    representative = {"frame-000000.png": 0, "frame-middle.png": max(0, expected_frames // 2), "frame-last.png": max(0, expected_frames - 1)}
    frame_results: dict[str, dict[str, Any]] = {}
    for filename, index in representative.items():
        frame_path = args.frames_dir / filename
        extracted = run_command(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(args.video),
                "-vf",
                f"select=eq(n\\,{index})",
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        frame_results[filename] = {
            "index": index,
            "path": filename,
            "returncode": extracted.returncode,
            "bytes": frame_path.stat().st_size if frame_path.exists() else 0,
        }

    result = {
        "status": "ok" if decode.returncode == 0 and decoded_frames == expected_frames and all(item["returncode"] == 0 and item["bytes"] > 0 for item in frame_results.values()) else "error",
        "video": args.video.name,
        "video_bytes": args.video.stat().st_size,
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "ffprobe_returncode": probe.returncode,
        "ffmpeg_full_decode_returncode": decode.returncode,
        "ffmpeg_full_decode_stderr": decode.stderr[-1200:],
        "decoded_video_frames": decoded_frames,
        "expected_video_frames_from_episode_json": expected_frames,
        "frame_count_matches": decoded_frames == expected_frames,
        "ffprobe": {
            "codec_name": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pix_fmt": stream.get("pix_fmt"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "duration": probe_json.get("format", {}).get("duration"),
            "nb_frames": stream.get("nb_frames"),
            "nb_read_frames": stream.get("nb_read_frames"),
        },
        "episode_terminal_reason": episode["episode"]["terminal_reason"],
        "episode_has_won": episode["episode"]["has_won"],
        "representative_frames": frame_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
