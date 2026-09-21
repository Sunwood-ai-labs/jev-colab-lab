"""T4-only limited teacher fine-tune for the Kev JevDash adapter.

The teacher is the fixed JevDash reflex policy copied for a separate control
audit: raw ``right_run`` on clear terrain and ``right_run_jump`` when the
published hazard/stall conditions require it.  This is deliberately recorded
as a game-specific teacher experiment, not as evidence that the original Kev
checkpoint understood the game.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


GAME_COMMIT = "eb2f92617bab5d5021a5e3cf5ef2bdaf8207d480"
GAME_REPO = "https://github.com/Sunwood-ai-labs/jevdash.git"
GAME_DIR = Path("/content/kev-jevdash-contract/jevdash")
SEED = 42
MAX_FRAMES = 1800
FPS = 60
DECISION_INTERVAL = 8
START_OFFSETS = (0, -16, 16)
TRAIN_OFFSETS = (0, -16)
VALIDATION_OFFSETS = (16,)
TEACHER_POLICY = "hazard_12_macro"
ACTION_ORDER = ("noop", "right", "right_run", "right_jump", "right_run_jump", "jump", "left")
QUESTION = (
    "Choose exactly one action macro for the next 8 simulation frames to reach the goal flag to the right. "
    "On clear ground, keep advancing with right_run. If a pipe, gap, or enemy is ahead and the player is "
    "grounded, choose right_run_jump early enough to clear it. While airborne, keep moving right and do not "
    "expect jump to create a second jump; grounded and airborne_frames describe jump phase. "
    "Telemetry obstacle_distance_tiles and gap_distance_tiles are coarse column distances from the player's "
    "tile, not exact front-edge collision clearance. Use enemy distance and vertical position together. "
    "Treat stalled_frames >= 3 as a failure to make forward progress and recover by moving right. "
    "Use left only for explicit backtracking in the supplied state. Return no prose."
)
ACTION_DESCRIPTIONS = {
    "noop": "Apply no directional or jump input. On the ground this decelerates horizontal speed; in the air it preserves horizontal inertia.",
    "right": "Move right: walking speed while grounded, but running horizontal speed while airborne; do not start a jump.",
    "right_run": "Move right at running speed across clear flat ground; do not start a jump.",
    "right_jump": "Move right and start a normal jump with upward velocity -13.5; an airborne jump input cannot create a second jump.",
    "right_run_jump": "Move right at running speed and start the stronger running jump with upward velocity -15.5 over an immediate pipe, gap, or enemy; an airborne jump input cannot create a second jump.",
    "jump": "Start a vertical normal jump with upward velocity -13.5 without intentional horizontal direction; an airborne jump input cannot create a second jump.",
    "left": "Move left away from the goal; use only when backtracking is explicitly necessary.",
}
RESULT_PATH = Path(os.environ.get("KEV_TRAIN_RESULT_PATH", "/content/kev-jevdash-teacher-training.json"))
DATASET_PATH = Path(os.environ.get("KEV_TRAIN_DATASET_PATH", "/content/kev-jevdash-teacher-dataset.json"))
TUNED_DIR = Path(os.environ.get("KEV_TUNED_DIR", "/content/kev-jevdash-tuned"))


def run_command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clone_game() -> None:
    if not (GAME_DIR / ".git").exists():
        GAME_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", GAME_REPO, str(GAME_DIR)], check=True)
    observed = run_command(["git", "rev-parse", "HEAD"], cwd=GAME_DIR)
    if observed != GAME_COMMIT:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", GAME_COMMIT], cwd=GAME_DIR, check=True)
        subprocess.run(["git", "switch", "--detach", GAME_COMMIT], cwd=GAME_DIR, check=True)
    if run_command(["git", "rev-parse", "HEAD"], cwd=GAME_DIR) != GAME_COMMIT:
        raise RuntimeError("game revision mismatch")
    if run_command(["git", "status", "--porcelain"], cwd=GAME_DIR):
        raise RuntimeError("game checkout is dirty")


def compact_state(full: dict[str, Any]) -> dict[str, Any]:
    nearest = full["hazard"].get("nearest_enemy")
    return {
        "mission": "Reach the goal flag to the right without dying.",
        "player": {key: full["player"][key] for key in ("x", "y", "vx", "vy", "grounded", "jumping", "airborne_frames", "running")},
        "progress": {key: full["episode"][key] for key in ("progress_pixels", "goal_distance_pixels", "stalled_frames", "score", "coins", "lives")},
        "hazard": {
            "enemy_ahead": full["hazard"]["enemy_ahead"],
            "nearest_enemy": nearest,
            "jump_must_start_now": full["hazard"]["jump_must_start_now"],
            "in_danger_zone": full["hazard"]["in_danger_zone"],
        },
        "terrain": full["terrain"],
        "local_radar": full["local_grid"],
    }


def teacher_action(obs: Any) -> tuple[str, list[str]]:
    """8-frame-safe macro teacher, verified separately before training.

    The published reflex threshold (2.2 tiles) is too late when held for
    eight frames.  This teacher starts a running jump when a coarse telemetry
    hazard is within twelve tile columns, while retaining right_run on clear
    ground.  It is a bounded game-specific teacher, not a learned policy.
    """

    terrain, hazard, player = obs.terrain, obs.hazard, obs.player
    reasons: list[str] = []
    obstacle_distance = terrain.obstacle_distance_tiles or 99
    gap_distance = terrain.gap_distance_tiles or 99
    enemy_distance = hazard.nearest_enemy.distance_pixels if hazard.nearest_enemy else 999
    if obstacle_distance <= 12:
        reasons.append("obstacle_within_12_tiles")
    if gap_distance <= 12:
        reasons.append("gap_critical")
    if enemy_distance <= 360:
        reasons.append("enemy_within_360_pixels")
    if hazard.jump_must_start_now:
        reasons.append("jump_must_start_now")
    if obs.episode.stalled_frames >= 3:
        reasons.append("stalled")
    if reasons:
        return "right_run_jump", reasons
    return "right_run", []


def step_game(player: Any, level: Any, action: str) -> None:
    player.apply_action(action)
    player.update_physics(level.tiles)
    for enemy in level.enemies:
        enemy.update(level.tiles)
        if enemy.alive and player.rect.colliderect(enemy.rect):
            feet_y = player.y + player.height
            if feet_y <= enemy.y + enemy.height * 0.65:
                enemy.stomp()
                player.vy = -11.0
                player.score += 100
            else:
                player.is_dead = True
    for coin in level.coins:
        if not coin.collected and player.rect.colliderect(coin.rect):
            coin.collected = True
            player.coins += 1
            player.score += 50
    if level.goal and player.rect.colliderect(level.goal.rect):
        player.has_won = True


def collect_dataset(Level: Any, Player: Any, Extractor: Any) -> list[dict[str, Any]]:
    import pygame

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    pygame.init()
    rows: list[dict[str, Any]] = []
    for start_offset in START_OFFSETS:
        random.seed(SEED)
        level = Level(1)
        player = Player(level.start_pos[0] + start_offset, level.start_pos[1])
        previous_action = None
        for frame in range(MAX_FRAMES):
            obs = Extractor.extract(player, level)
            action, reasons = teacher_action(obs)
            # The teacher is itself evaluated only at the serving cadence and
            # its selected macro is held for all eight physical frames.
            if frame % DECISION_INTERVAL == 0:
                full = obs.model_dump(mode="json")
                rows.append(
                    {
                        "start_offset_pixels": start_offset,
                        "frame": frame,
                        "state": compact_state(full),
                        "target_action": action,
                        "teacher_reasons": reasons,
                    }
                )
            previous_action = action
            step_game(player, level, action)
            if player.is_dead or player.has_won:
                break
    pygame.quit()
    return rows


def build_record(SystemOneRequest: Any, to_record: Any, state: dict[str, Any], order: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "state": state,
        "model": "kev-0.5b",
        "questions": {
            "action": {
                "type": "choice",
                "instructions": QUESTION,
                "criteria": {key: ACTION_DESCRIPTIONS[key] for key in order},
            }
        },
    }
    validated = SystemOneRequest.model_validate(request)
    record, meta = to_record(validated)
    if meta[0].get("keys") != list(order):
        raise RuntimeError("teacher record mapping mismatch")
    return record, meta[0]


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("torch", "transformers", "peft", "pygame", "pydantic"):
        try:
            module = importlib.import_module(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def main() -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "jev-colab-lab/kev-jevdash-teacher-training-v1",
        "status": "started",
        "game_commit": GAME_COMMIT,
        "seed": SEED,
        "fps": FPS,
        "decision_interval_frames": DECISION_INTERVAL,
        "teacher": (
            "8-frame-safe hazard_12_macro: right_run on clear ground, right_run_jump when coarse obstacle/gap "
            "is within 12 tiles, enemy within 360 pixels, jump_must_start_now, or stalled; game-specific limited "
            "teacher, not the original Kev model"
        ),
        "representation": "compact_json",
        "question_variant": "rules_v2",
        "candidate_actions": list(ACTION_ORDER),
        "train_offsets": list(TRAIN_OFFSETS),
        "validation_offsets": list(VALIDATION_OFFSETS),
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": package_versions()},
        "hardware": {},
        "training": {},
        "dataset": {},
        "error": None,
    }
    phase = "prepare"
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        gpu_name = torch.cuda.get_device_name(0)
        if "T4" not in gpu_name.upper():
            raise RuntimeError("requested T4 but a different GPU was reported")
        result["hardware"] = {
            "name": gpu_name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 3),
            "torch_cuda": torch.version.cuda,
        }
        clone_game()
        sys.path.insert(0, "/content")
        import t4_inference

        source = t4_inference.clone_upstream()
        model_sources = t4_inference.download_models()
        sys.path.insert(0, str(t4_inference.UPSTREAM_DIR))
        from kev.api import SystemOneRequest, to_record
        from kev.model import DecisionModel, encode, load_tokenizer
        from peft import PeftModel
        from jev_platformer.engine.entities import Player
        from jev_platformer.engine.world import Level
        from jev_platformer.telemetry.extractor import TelemetryExtractor

        phase = "collect_teacher_dataset"
        dataset_rows = collect_dataset(Level, Player, TelemetryExtractor)
        dataset_json = {
            "schema": "jev-colab-lab/kev-jevdash-teacher-dataset-v1",
            "game_commit": GAME_COMMIT,
            "teacher": result["teacher"],
            "representation": "compact_json",
            "question_variant": "rules_v2",
            "rows": dataset_rows,
        }
        DATASET_PATH.write_text(json.dumps(dataset_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        counts = Counter(row["target_action"] for row in dataset_rows)
        result["dataset"] = {
            "rows": len(dataset_rows),
            "sha256": sha256_json(dataset_json),
            "target_action_counts": dict(counts),
            "start_offsets": list(START_OFFSETS),
            "dataset_path_remote": str(DATASET_PATH),
        }

        phase = "load_official_adapter"
        tokenizer = load_tokenizer(str(t4_inference.BASE_DIR))
        model = DecisionModel(str(t4_inference.BASE_DIR), tokenizer, "cuda", lora=None)
        model.lm = PeftModel.from_pretrained(model.lm, str(t4_inference.ADAPTER_DIR), is_trainable=True).to("cuda")
        head_meta = torch.load(t4_inference.ADAPTER_DIR / "head.pt", map_location="cpu", weights_only=False)
        model.head.load_state_dict(head_meta["head"])
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
        rng = random.Random(SEED)
        natural_train_rows = [row for row in dataset_rows if row["start_offset_pixels"] in TRAIN_OFFSETS]
        validation_rows = [row for row in dataset_rows if row["start_offset_pixels"] in VALIDATION_OFFSETS]
        natural_counts = Counter(row["target_action"] for row in natural_train_rows)
        run_rows = [row for row in natural_train_rows if row["target_action"] == "right_run"]
        jump_rows = [row for row in natural_train_rows if row["target_action"] == "right_run_jump"]
        if not run_rows or not jump_rows:
            raise RuntimeError("teacher dataset does not contain both right_run and right_run_jump labels")
        # The safe macro teacher is intentionally conservative, so its natural
        # labels are jump-heavy.  Reweight only the clear-ground class by
        # deterministic oversampling; validation remains the untouched offset.
        target_run_count = max(len(run_rows), round(len(jump_rows) * 0.45 / 0.55))
        balanced_train_rows = list(natural_train_rows)
        for index in range(target_run_count - len(run_rows)):
            balanced_train_rows.append(run_rows[index % len(run_rows)])
        epochs = 3
        accumulation = 4
        max_updates = 600
        result["training"] = {
            "initial_adapter": model_sources["adapter"]["revision"],
            "base_revision": model_sources["base"]["revision"],
            "epochs_requested": epochs,
            "gradient_accumulation": accumulation,
            "max_optimizer_updates": max_updates,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "train_rows_natural": len(natural_train_rows),
            "train_rows_balanced": len(balanced_train_rows),
            "validation_rows": len(validation_rows),
            "natural_train_action_counts": dict(natural_counts),
            "balanced_train_action_counts": dict(Counter(row["target_action"] for row in balanced_train_rows)),
            "clear_ground_oversampling": "right_run rows repeated to approximate a 45/55 right_run/right_run_jump mix; no validation oversampling",
            "random_option_permutation": True,
            "loss": "cross_entropy on teacher action; no game reward or fallback",
        }

        phase = "fine_tune"
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        update_count = 0
        loss_history: list[float] = []
        for epoch in range(epochs):
            rng.shuffle(balanced_train_rows)
            batch_losses: list[float] = []
            for index, row in enumerate(balanced_train_rows):
                order = tuple(rng.sample(list(ACTION_ORDER), len(ACTION_ORDER)))
                record, meta = build_record(SystemOneRequest, to_record, row["state"], order)
                logits = model(encode(tokenizer, record))[0]
                target_index = order.index(row["target_action"])
                target = torch.tensor([target_index], device="cuda")
                loss = torch.nn.functional.cross_entropy(logits[None], target)
                (loss / accumulation).backward()
                batch_losses.append(float(loss.detach().cpu()))
                if (index + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_count += 1
                    if update_count >= max_updates:
                        break
            if batch_losses and update_count % accumulation:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            loss_history.append(sum(batch_losses) / len(batch_losses) if batch_losses else 0.0)
            if update_count >= max_updates:
                break

        def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
            model.eval()
            correct = 0
            predictions: Counter[str] = Counter()
            target_counts: Counter[str] = Counter()
            with torch.inference_mode():
                for row in rows:
                    record, meta = build_record(SystemOneRequest, to_record, row["state"], ACTION_ORDER)
                    probs = model.probs(encode(tokenizer, record))[0].tolist()
                    prediction = ACTION_ORDER[max(range(len(probs)), key=lambda i: probs[i])]
                    predictions[prediction] += 1
                    target_counts[row["target_action"]] += 1
                    correct += int(prediction == row["target_action"])
            model.train()
            return {
                "rows": len(rows),
                "accuracy": correct / len(rows) if rows else 0.0,
                "predicted_action_counts": dict(predictions),
                "target_action_counts": dict(target_counts),
            }

        phase = "evaluate_and_save"
        result["training"].update(
            {
                "optimizer_updates": update_count,
                "mean_loss_per_epoch": loss_history,
                "train_eval_natural": evaluate(natural_train_rows),
                "train_eval_balanced": evaluate(balanced_train_rows),
                "validation_eval": evaluate(validation_rows),
            }
        )
        TUNED_DIR.mkdir(parents=True, exist_ok=True)
        model.lm.save_pretrained(TUNED_DIR)
        torch.save(
            {
                "head": model.head.state_dict(),
                "base": str(t4_inference.BASE_DIR),
                "teacher": result["teacher"],
                "training": result["training"],
            },
            TUNED_DIR / "head.pt",
        )
        tokenizer.save_pretrained(TUNED_DIR)
        torch.cuda.synchronize()
        result["hardware"]["peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 3)
        result["hardware"]["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 3)
        result["sources"] = {
            "game": {"repository": GAME_REPO, "commit": GAME_COMMIT},
            "kev": {key: source[key] for key in ("repository", "ref", "commit")},
            "adapter": {"repository": model_sources["adapter"]["repo"], "revision": model_sources["adapter"]["revision"]},
            "base": {"repository": model_sources["base"]["repo"], "revision": model_sources["base"]["revision"]},
        }
        result["tuned_model"] = {
            "remote_path": str(TUNED_DIR),
            "weights_committed": False,
            "served_by": "KevDecisionAdapter with KEV_ADAPTER_PATH",
        }
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": "Teacher fine-tune failed; exception text and paths omitted from sanitized result.",
        }
    result["wall_clock_seconds"] = time.perf_counter() - started
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "phase": phase, "result": str(RESULT_PATH)}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
