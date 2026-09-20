"""Create a display-only replay of a recorded JevDash episode.

The source video is the untouched Colab/T4 recording.  This helper does not
rerun the game or the model: it only redraws the two unqueried HUD labels and
re-encodes the same frame sequence for a clean visual deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


HUD_BG = "0x161b2a"
HUD_BORDER = "0x262d42"
HUD_TEXT = "0x94a3b8"
HUD_TEXT_X = 1048
HUD_PROGRESS_X = 1165
CARD_X = 900
CARD_Y = 88
CARD_W = 360
CARD_H = 68
LATENCY_X = 912
FONT_SIZE = 11
PROGRESS_FONT_SIZE = 22


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_sha256(metadata: dict[str, Any]) -> str:
    episode = dict(metadata.get("episode") or {})
    for derived_key in ("terminal_hold_frames_upstream_cli", "expected_video_frames"):
        episode.pop(derived_key, None)
    payload = {
        "seed": metadata.get("seed"),
        "level": metadata.get("level"),
        "episode": episode,
        "decisions": metadata.get("decisions", []),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames,duration,pix_fmt,codec_name",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def full_decode(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stderr": result.stderr,
    }


def write_representative_frames(path: Path) -> list[dict[str, Any]]:
    stream = (ffprobe(path).get("streams") or [{}])[0]
    frame_count = int(stream.get("nb_read_frames") or 0)
    if frame_count <= 0:
        raise RuntimeError(f"No decoded frames in {path}")
    specs = [("first", 0), ("middle", frame_count // 2), ("last", frame_count - 1)]
    records: list[dict[str, Any]] = []
    for name, frame_index in specs:
        frame_path = path.parent / f"{name}-presentation-replay.png"
        run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"select=eq(n\\,{frame_index})",
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        records.append(
            {
                "name": name,
                "frame": frame_index,
                "path": str(frame_path),
                "sha256": sha256_file(frame_path),
            }
        )
    return records


def find_font(*names: str) -> Path:
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
        for name in names
    ] + [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "consola.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "cour.ttf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("A monospace Windows font was not found for ffmpeg drawtext")


def filter_path(path: Path) -> str:
    # ffmpeg's filter parser treats the Windows drive colon as a separator.
    return str(path).replace("\\", "/").replace(":", r"\:")


def filter_text(value: str) -> str:
    return value.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")


def make_video(source: Path, output: Path, progress_text: str, decisions: list[dict[str, Any]]) -> None:
    font = filter_path(find_font("consola.ttf", "cour.ttf"))
    progress_font = filter_path(find_font("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf"))
    filters = [
        f"drawbox=x={CARD_X}:y={CARD_Y}:w={CARD_W}:h={CARD_H}:color={HUD_BG}:t=fill",
        f"drawbox=x={CARD_X}:y={CARD_Y}:w={CARD_W}:h={CARD_H}:color={HUD_BORDER}:t=1",
        (
            "drawtext="
            f"fontfile='{font}':text='{filter_text('INFERENCE DELAY')}':"
            f"fontcolor={HUD_TEXT}:fontsize={FONT_SIZE}:x={LATENCY_X}:y=98"
        ),
        (
            "drawtext="
            f"fontfile='{font}':text='{filter_text('DANGER: N/A')}':"
            f"fontcolor={HUD_TEXT}:fontsize={FONT_SIZE}:x={HUD_TEXT_X}:y=104"
        ),
        (
            "drawtext="
            f"fontfile='{font}':text='{filter_text('URGENCY: N/A')}':"
            f"fontcolor={HUD_TEXT}:fontsize={FONT_SIZE}:x={HUD_TEXT_X}:y=124"
        ),
        (
            "drawtext="
            f"fontfile='{font}':text='PROGRESS':fontcolor={HUD_TEXT}:"
            f"fontsize={FONT_SIZE}:x={HUD_PROGRESS_X}:y=98"
        ),
        (
            "drawtext="
            f"fontfile='{progress_font}':text='{filter_text(progress_text)}':"
            f"fontcolor=0xf8fafc:fontsize={PROGRESS_FONT_SIZE}:x={HUD_PROGRESS_X}:y=116"
        ),
    ]
    source_frames = int((ffprobe(source).get("streams") or [{}])[0].get("nb_read_frames") or 0)
    for index, decision in enumerate(decisions):
        start = int(decision["simulation_frame"])
        end = int(decisions[index + 1]["simulation_frame"]) - 1 if index + 1 < len(decisions) else source_frames - 1
        latency_text = filter_text(f"{float(decision['inference_ms']):.1f} ms")
        enable = f"between(n\\,{start}\\,{end})"
        filters.append(
            "drawtext="
            f"fontfile='{progress_font}':text='{latency_text}':fontcolor=0x38bdf8:"
            f"fontsize={PROGRESS_FONT_SIZE}:x={LATENCY_X}:y=116:enable='{enable}'"
        )
    filter_graph = ",".join(filters)
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            filter_graph,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "60",
            "-an",
            "-map_metadata",
            "-1",
            str(output),
        ]
    )


def replay_metadata(source_json: Path, source_video: Path, output_video: Path) -> dict[str, Any]:
    metadata = json.loads(source_json.read_text(encoding="utf-8"))
    source_probe = ffprobe(source_video)
    replay_probe = ffprobe(output_video)
    source_stream = (source_probe.get("streams") or [{}])[0]
    replay_stream = (replay_probe.get("streams") or [{}])[0]
    source_frames = int(source_stream.get("nb_read_frames") or 0)
    replay_frames = int(replay_stream.get("nb_read_frames") or 0)
    episode = metadata.setdefault("episode", {})
    simulation_frames = int(episode.get("simulation_frames") or 0)
    appended_terminal = int(episode.get("terminal_hold_frames_appended_by_adapter") or 0)
    upstream_hold = source_frames - simulation_frames - appended_terminal
    episode["terminal_hold_frames_upstream_cli"] = upstream_hold
    episode["expected_video_frames"] = source_frames

    metadata["replay"] = {
        "type": "display_only_replay",
        "source": "Colab T4 MP4 retained as kev-jevdash-level1-colab-raw.mp4",
        "physics_and_model_unchanged": True,
        "frame_sequence_unchanged": source_frames == replay_frames,
        "hud_patch": "redrew the complete metrics card with latency, DANGER/URGENCY N/A, and PROGRESS in separate columns",
        "trajectory_sha256": trajectory_sha256(metadata),
        "terminal_frame_breakdown": {
            "simulation_frames": simulation_frames,
            "upstream_death_or_clear_hold_frames": upstream_hold,
            "adapter_terminal_hold_frames": appended_terminal,
            "total_source_frames": source_frames,
            "total_replay_frames": replay_frames,
        },
    }
    metadata["video"] = {
        **metadata.get("video", {}),
        "path": str(output_video),
        "sha256": sha256_file(output_video),
        "frames_from_replay": replay_frames,
        "expected_frames": source_frames,
        "source_colab_video": str(source_video),
        "source_colab_sha256": sha256_file(source_video),
        "source_colab_frames": source_frames,
        "representative_frames": write_representative_frames(output_video),
        "ffprobe_decode": {
            "probe": replay_probe,
            "full_decode": full_decode(output_video),
        },
    }
    return metadata


def write_manifest(
    source_json: Path,
    source_video: Path,
    replay_json: Path,
    replay_video: Path,
    replay_metadata_value: dict[str, Any],
    manifest_path: Path,
) -> None:
    source_metadata = json.loads(source_json.read_text(encoding="utf-8"))
    source_probe = ffprobe(source_video)
    replay_probe = ffprobe(replay_video)
    source_stream = (source_probe.get("streams") or [{}])[0]
    replay_stream = (replay_probe.get("streams") or [{}])[0]
    source_trajectory = trajectory_sha256(source_metadata)
    replay_trajectory = trajectory_sha256(replay_metadata_value)
    manifest = {
        "type": "jevdash_video_manifest",
        "original_colab": {
            "path": str(source_video),
            "sha256": sha256_file(source_video),
            "frames": int(source_stream.get("nb_read_frames") or 0),
            "fps": source_stream.get("r_frame_rate"),
            "duration_s": float(source_stream.get("duration") or 0.0),
        },
        "presentation_replay": {
            "path": str(replay_video),
            "sha256": sha256_file(replay_video),
            "frames": int(replay_stream.get("nb_read_frames") or 0),
            "fps": replay_stream.get("r_frame_rate"),
            "duration_s": float(replay_stream.get("duration") or 0.0),
            "full_decode": full_decode(replay_video),
        },
        "trajectory": {
            "original_colab_sha256": source_trajectory,
            "presentation_replay_sha256": replay_trajectory,
            "equal": source_trajectory == replay_trajectory,
        },
        "state_match": {
            "status_equal": source_metadata.get("status") == replay_metadata_value.get("status"),
            "simulation_frames_equal": source_metadata.get("episode", {}).get("simulation_frames")
            == replay_metadata_value.get("episode", {}).get("simulation_frames"),
            "decision_count_equal": source_metadata.get("episode", {}).get("decision_count")
            == replay_metadata_value.get("episode", {}).get("decision_count"),
            "physics_and_model_unchanged": replay_metadata_value["replay"]["physics_and_model_unchanged"],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    if not args.source_video.is_file():
        raise FileNotFoundError(args.source_video)
    if not args.source_json.is_file():
        raise FileNotFoundError(args.source_json)
    source_metadata = json.loads(args.source_json.read_text(encoding="utf-8"))
    progress_text = f"{int(source_metadata['episode']['progress_pixels'])}px"
    make_video(args.source_video, args.output_video, progress_text, source_metadata.get("decisions", []))
    metadata = replay_metadata(args.source_json, args.source_video, args.output_video)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        args.source_json,
        args.source_video,
        args.output_json,
        args.output_video,
        metadata,
        args.output_manifest,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
