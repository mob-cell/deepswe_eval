#!/usr/bin/env python3
"""Record per-turn token usage from finished or in-progress Pier trajectories.

Reads agent/mini-swe-agent.trajectory.json under a job directory (or a directory
of jobs). Each model call is one turn. The numbers are the usage object returned
for that call, not an estimate.

Task and job totals sum those per-turn values. prompt_tokens on each turn already
includes the full context sent for that request, so the task total is the sum of
what the API reported.

Router responses also carry slm_prompt_tokens, slm_completion_tokens,
llm_prompt_tokens, and llm_completion_tokens. Those are summed per task so the
small model and the large model can be priced separately. offload_turns counts
turns that called the large model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAJECTORY_NAME = "mini-swe-agent.trajectory.json"
TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "completion_text_tokens",
    "total_tokens",
)
# Present when the offload adapter returned a per-model split.
# prompt_tokens / completion_tokens on those turns are the sum of both models.
SPLIT_FIELDS = (
    "slm_prompt_tokens",
    "slm_completion_tokens",
    "llm_prompt_tokens",
    "llm_completion_tokens",
)


def int_or_none(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def details(usage, *keys):
    for key in keys:
        value = usage.get(key)
        if isinstance(value, dict):
            return value
    return {}


def split_from_usage(usage):
    """Small-model and large-model counts, when the adapter recorded both."""
    empty = {field: None for field in SPLIT_FIELDS}
    empty["offloaded"] = None
    if not isinstance(usage, dict):
        return empty
    if "slm_prompt_tokens" not in usage and "llm_prompt_tokens" not in usage:
        return empty
    values = {}
    for field in SPLIT_FIELDS:
        number = int_or_none(usage.get(field))
        values[field] = 0 if number is None else number
    values["offloaded"] = int(values["llm_prompt_tokens"] + values["llm_completion_tokens"] > 0)
    return values


def normalize_usage(usage):
    prompt = int_or_none(usage.get("prompt_tokens"))
    if prompt is None:
        prompt = int_or_none(usage.get("input_tokens"))
    completion = int_or_none(usage.get("completion_tokens"))
    if completion is None:
        completion = int_or_none(usage.get("output_tokens"))
    total = int_or_none(usage.get("total_tokens"))
    prompt_details = details(usage, "prompt_tokens_details", "input_tokens_details")
    completion_details = details(usage, "completion_tokens_details", "output_tokens_details")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": int_or_none(completion_details.get("reasoning_tokens")),
        "cached_tokens": int_or_none(prompt_details.get("cached_tokens")),
        "completion_text_tokens": int_or_none(completion_details.get("text_tokens")),
        "total_tokens": total,
    }


def parse_response(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_call(message):
    """Return one model call when the message carries an API usage object.

    Successful calls are assistant/response messages. Format-error calls are stored
    on the following user message, with the raw response under extra.response.
    """
    if isinstance(message.get("usage"), dict):
        return {
            "usage": message["usage"],
            "model": message.get("model"),
            "response_id": message.get("id"),
            "created_at": message.get("created_at", message.get("created")),
            "source": "message.usage",
            "message_role": message.get("role") or message.get("object"),
        }

    extra = message.get("extra") if isinstance(message.get("extra"), dict) else None
    response = parse_response(extra.get("response")) if extra else None
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return None

    created = response.get("created", response.get("created_at"))
    if created is None:
        created = extra.get("timestamp")
    return {
        "usage": response["usage"],
        "model": response.get("model") or message.get("model"),
        "response_id": response.get("id") or message.get("id"),
        "created_at": created,
        "source": "extra.response.usage",
        "message_role": message.get("role") or message.get("object"),
    }


def trial_task_name(trial_dir_name):
    task, sep, _suffix = trial_dir_name.rpartition("__")
    if sep and task:
        return task
    return trial_dir_name


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def turns_from_trajectory(path, job_name):
    payload = load_json(path)
    info = payload.get("info") or {}
    config = info.get("config") or {}
    model_cfg = config.get("model") or {}
    configured_model = model_cfg.get("model_name")
    exit_status = info.get("exit_status") or ""
    trial_dir = path.parent.parent
    trial = trial_dir.name
    turns = []
    for index, message in enumerate(payload.get("messages") or []):
        if not isinstance(message, dict):
            continue
        call = extract_call(message)
        if call is None:
            continue
        usage = call["usage"] or {}
        tokens = normalize_usage(usage) if call["usage"] else {field: None for field in TOKEN_FIELDS}
        split = split_from_usage(call["usage"])
        turns.append(
            {
                "job": job_name,
                "trial": trial,
                "task": trial_task_name(trial),
                "turn": len(turns) + 1,
                "message_index": index,
                "model": call["model"] or configured_model,
                "configured_model": configured_model,
                "message_role": call["message_role"],
                "response_id": call["response_id"],
                "created_at": call["created_at"],
                "source": call["source"],
                **tokens,
                **split,
                "usage": call["usage"],
            }
        )
    return {
        "job": job_name,
        "trial": trial,
        "task": trial_task_name(trial),
        "configured_model": configured_model,
        "exit_status": exit_status,
        "complete": bool(exit_status),
        "trajectory": str(path),
        "api_calls_reported": (info.get("model_stats") or {}).get("api_calls"),
        "turns": turns,
    }


def sum_field(turns, field):
    values = [turn[field] for turn in turns if turn[field] is not None]
    if not values:
        return None
    return sum(values)


def summarize_task(task):
    turns = task["turns"]
    missing = sum(1 for turn in turns if turn["usage"] is None)
    summary = {
        "job": task["job"],
        "trial": task["trial"],
        "task": task["task"],
        "model": next((turn["model"] for turn in turns if turn["model"]), task["configured_model"]),
        "configured_model": task["configured_model"],
        "complete": task["complete"],
        "exit_status": task["exit_status"],
        "n_turns": len(turns),
        "n_turns_missing_usage": missing,
        "api_calls_reported": task["api_calls_reported"],
        "trajectory": task["trajectory"],
    }
    for field in (*TOKEN_FIELDS, *SPLIT_FIELDS):
        summary[field] = sum_field(turns, field)
    summary["offload_turns"] = sum_field(turns, "offloaded")
    return summary


def summarize_jobs(tasks):
    def rollup(rows):
        rolled = {"n_tasks": len(rows), "n_turns": sum(row["n_turns"] for row in rows)}
        for field in (*TOKEN_FIELDS, *SPLIT_FIELDS, "offload_turns"):
            rolled[field] = sum_field(rows, field)
        return rolled

    completed = [row for row in tasks if row["complete"]]
    return {
        "tasks": rollup(tasks),
        "completed_tasks": rollup(completed),
    }


def find_trajectories(root):
    return sorted(path for path in root.rglob(TRAJECTORY_NAME) if path.is_file())


def group_by_job(trajectories, root):
    groups = {}
    root = root.resolve()
    for path in trajectories:
        trial_dir = path.parent.parent
        job_dir = trial_dir.parent
        if job_dir == root or root in job_dir.parents:
            job_name = job_dir.name
        else:
            job_name = root.name
            job_dir = root
        groups.setdefault((job_name, job_dir), []).append(path)
    return groups


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_job(job_name, trajectories, out_dir):
    tasks = []
    errors = []
    for path in trajectories:
        try:
            tasks.append(turns_from_trajectory(path, job_name))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"trajectory": str(path), "error": str(exc)})
    tasks.sort(key=lambda item: item["trial"])
    task_summaries = [summarize_task(task) for task in tasks]
    turns = [turn for task in tasks for turn in task["turns"]]
    report_dir = out_dir / job_name
    write_jsonl(report_dir / "turns.jsonl", turns)
    summary = {
        "job": job_name,
        "aggregation": (
            "Each turn stores the usage object from that model call. "
            "Task totals are the sum of those per-turn values. "
            "When slm_* and llm_* are present, they are that turn's small-model and "
            "large-model counts, and prompt_tokens / completion_tokens are their sums. "
            "offload_turns is how many turns called the large model. "
            "Older trajectories without the split leave slm_* / llm_* null."
        ),
        "n_trajectories": len(tasks),
        "errors": errors,
        **summarize_jobs(task_summaries),
        "task_list": task_summaries,
    }
    write_json(report_dir / "summary.json", summary)
    return summary


def print_summary(summary):
    tasks = summary["tasks"]
    completed = summary["completed_tasks"]
    print(
        f"{summary['job']}: trajectories={summary['n_trajectories']} "
        f"turns={tasks['n_turns']} total_tokens={tasks['total_tokens']} "
        f"prompt_tokens={tasks['prompt_tokens']} completion_tokens={tasks['completion_tokens']} "
        f"slm_prompt={tasks['slm_prompt_tokens']} slm_completion={tasks['slm_completion_tokens']} "
        f"llm_prompt={tasks['llm_prompt_tokens']} llm_completion={tasks['llm_completion_tokens']} "
        f"offload_turns={tasks['offload_turns']} "
        f"reasoning_tokens={tasks['reasoning_tokens']} "
        f"completed_tasks={completed['n_tasks']}/{tasks['n_tasks']}"
    )
    if summary["errors"]:
        print(f"  errors: {len(summary['errors'])}", file=sys.stderr)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="A job directory, a trial directory, or a directory that contains jobs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for per-job turns.jsonl and summary.json",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    wrote = 0
    for raw_path in args.paths:
        root = raw_path.expanduser().resolve()
        if not root.exists():
            print(f"missing path: {root}", file=sys.stderr)
            return 1
        trajectories = find_trajectories(root)
        if not trajectories:
            print(f"no {TRAJECTORY_NAME} under {root}", file=sys.stderr)
            continue
        for (job_name, _job_dir), paths in sorted(group_by_job(trajectories, root).items()):
            summary = collect_job(job_name, paths, args.out)
            print_summary(summary)
            print(f"  wrote {args.out / job_name}")
            wrote += 1
    return 0 if wrote else 1


if __name__ == "__main__":
    raise SystemExit(main())
