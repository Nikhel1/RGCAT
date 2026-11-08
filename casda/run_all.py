#!/usr/bin/env python
# coding: utf-8
"""
casda/run_all.py — CASDA data download orchestrator
=====================================================

Runs the five casda/ data-download steps in order:

  Step 1  radio_image_download_casda.py   EMU FITS tiles      → data/images/
  Step 2  catalog_download_casda.py       selavy CSVs         → data/catalogs/
  Step 3  wise_mosaics.sh                 unWISE W1 mosaics   → data/images_wise/
  Step 4  download_catwise_allwise_catalogs.py  CatWISE/AllWISE → data/catalogs_wise/
  Step 5  download_desi_catalogs.py       DESI DR10 + photo-z → data/catalogs_desi/

Features
--------
- Prompts for CASDA credentials once; passes them to child processes via
  environment variables (CASDA_USER / CASDA_PASS) so no re-prompting.
- Writes a per-step log file to logs/casda/<step>_<timestamp>.log.
- Maintains a JSON state file (logs/casda/state.json) so completed steps
  are skipped on re-run.
- Prints a live progress dashboard between steps.
- Exits non-zero if any required step fails.

Usage (from repo root):
    python casda/run_all.py [options]

Options:
    --config PATH        pipeline.yaml path  (default: config/pipeline.yaml)
    --steps N [N ...]    run only these step numbers, e.g. --steps 1 3 5
    --from-step N        start from step N, skip earlier ones
    --force              re-run steps even if already marked complete
    --no-credentials     skip CASDA credential prompt (steps 1 & 2 will fail
                         unless CASDA_USER/CASDA_PASS are already set)
    --dry-run            print what would run without executing anything
"""

import argparse
import getpass
import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = [
    {
        "id": 1,
        "name": "radio_images",
        "label": "EMU radio FITS tiles",
        "cmd": [sys.executable, "casda/radio_image_download_casda.py"],
        "needs_casda_auth": True,
    },
    {
        "id": 2,
        "name": "selavy_catalogs",
        "label": "Selavy component catalogues",
        "cmd": [sys.executable, "casda/catalog_download_casda.py"],
        "needs_casda_auth": True,
    },
    {
        "id": 3,
        "name": "wise_mosaics",
        "label": "unWISE W1 mosaics",
        "cmd": ["bash", "casda/wise_mosaics.sh"],
        "needs_casda_auth": False,
        "config_flag": "--config",   # wise_mosaics.sh accepts --config
    },
    {
        "id": 4,
        "name": "catwise_allwise",
        "label": "CatWISE-2020 + AllWISE catalogs",
        "cmd": [sys.executable, "casda/download_catwise_allwise_catalogs.py"],
        "needs_casda_auth": False,
        "config_flag": "--config",
    },
    {
        "id": 5,
        "name": "desi_catalogs",
        "label": "DESI DR10 photometry + photo-z",
        "cmd": [sys.executable, "casda/download_desi_catalogs.py"],
        "needs_casda_auth": False,
        "config_flag": "--config",
    },
]

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return {}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2))


def mark_complete(state_file: Path, state: dict, step_name: str,
                  duration_s: float) -> None:
    state[step_name] = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 1),
    }
    save_state(state_file, state)


def mark_failed(state_file: Path, state: dict, step_name: str,
                exit_code: int, duration_s: float) -> None:
    state[step_name] = {
        "status": "failed",
        "exit_code": exit_code,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 1),
    }
    save_state(state_file, state)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def _c(text, *codes):
    """Wrap text in ANSI codes if stdout is a terminal."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


def print_banner():
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║          RGCAT · CASDA Data Download Suite           ║",
        "╚══════════════════════════════════════════════════════╝",
    ]
    print(_c("\n".join(lines), BOLD, CYAN))
    print()


def print_plan(steps_to_run: list, state: dict):
    print(_c("Steps to execute:", BOLD))
    for step in STEPS:
        sid = step["id"]
        name = step["name"]
        s = state.get(name, {})
        status = s.get("status", "pending")

        if step not in steps_to_run:
            tag = _c("  SKIP", DIM)
        elif status == "complete":
            tag = _c("  DONE", GREEN)
        elif status == "failed":
            tag = _c("  FAIL", RED)
        else:
            tag = _c("   RUN", YELLOW)

        label = step["label"]
        dur = f"  ({s['duration_s']:.0f}s)" if "duration_s" in s else ""
        print(f"  [{sid}] {tag}  {label}{_c(dur, DIM)}")
    print()


def print_step_header(step: dict):
    bar = "─" * 56
    print(_c(f"\n┌{bar}┐", CYAN))
    print(_c(f"│  Step {step['id']}/5 · {step['label']:<46}│", CYAN, BOLD))
    print(_c(f"└{bar}┘", CYAN))


def print_summary(results: list, total_s: float):
    print(_c("\n══════════════════ Summary ══════════════════", BOLD))
    all_ok = True
    for r in results:
        icon = _c("✓", GREEN) if r["ok"] else _c("✗", RED)
        dur  = _c(f"  {r['duration_s']:.0f}s", DIM)
        print(f"  {icon}  [{r['id']}] {r['label']}{dur}")
        if not r["ok"]:
            all_ok = False
    print(_c(f"\n  Total time: {total_s:.0f}s", DIM))
    if all_ok:
        print(_c("\n  All steps completed successfully.\n", BOLD, GREEN))
    else:
        print(_c("\n  Some steps failed — check logs/casda/ for details.\n",
                  BOLD, RED))
    return all_ok


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(step: dict, config: str, log_dir: Path, env: dict,
             dry_run: bool) -> tuple[bool, float]:
    """
    Execute one step.  Returns (success: bool, duration_s: float).
    Tees output to both the terminal and a log file.
    """
    cmd = list(step["cmd"])
    if "config_flag" in step:
        cmd += [step["config_flag"], config]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = log_dir / f"{step['name']}_{timestamp}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(_c(f"  Command : ", DIM) + " ".join(cmd))
    print(_c(f"  Log     : ", DIM) + str(log_path))
    print()

    if dry_run:
        print(_c("  [dry-run] skipping execution.", YELLOW))
        return True, 0.0

    t0 = time.monotonic()
    try:
        with open(log_path, "w") as log_fh:
            # Write a header to the log
            log_fh.write(f"# Step: {step['name']}\n")
            log_fh.write(f"# Command: {' '.join(cmd)}\n")
            log_fh.write(f"# Started: {datetime.now().isoformat()}\n")
            log_fh.write(f"# {'─' * 60}\n\n")
            log_fh.flush()

            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Tee: write each line to terminal and log simultaneously
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fh.write(line)

            proc.wait()

            log_fh.write(f"\n# {'─' * 60}\n")
            log_fh.write(f"# Exit code: {proc.returncode}\n")
            log_fh.write(f"# Finished: {datetime.now().isoformat()}\n")

    except KeyboardInterrupt:
        print(_c("\n  Interrupted by user.", YELLOW))
        return False, time.monotonic() - t0

    duration = time.monotonic() - t0
    success  = proc.returncode == 0
    return success, duration


# ---------------------------------------------------------------------------
# Credential helper
# ---------------------------------------------------------------------------

def prompt_casda_credentials() -> tuple[str, str]:
    """Prompt for OPAL credentials once and return (username, password)."""
    print(_c("CASDA / OPAL credentials", BOLD))
    print("  Steps 1 and 2 require authentication.")
    print("  Credentials are passed to child processes via environment")
    print("  variables and are never written to disk.\n")
    username = input("  CASDA (OPAL) username: ").strip()
    password = getpass.getpass("  CASDA (OPAL) password: ")
    print()
    return username, password


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            RGCAT CASDA data download orchestrator.
            Runs all five casda/ download steps in sequence.
        """),
    )
    parser.add_argument(
        "--config", default="config/pipeline.yaml",
        help="Path to pipeline.yaml (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", metavar="N",
        help="Run only these step numbers, e.g. --steps 1 3 5",
    )
    parser.add_argument(
        "--from-step", type=int, metavar="N",
        help="Start from step N, skipping earlier steps",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if already marked complete in state file",
    )
    parser.add_argument(
        "--no-credentials", action="store_true",
        help="Skip CASDA credential prompt "
             "(CASDA_USER / CASDA_PASS must already be set in the environment)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing anything",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    log_dir    = repo_root / "logs" / "casda"
    state_file = log_dir / "state.json"
    state      = load_state(state_file)

    # Determine which steps to run
    if args.steps:
        step_ids = set(args.steps)
    elif args.from_step:
        step_ids = {s["id"] for s in STEPS if s["id"] >= args.from_step}
    else:
        step_ids = {s["id"] for s in STEPS}

    steps_to_run = [s for s in STEPS if s["id"] in step_ids]

    # Skip already-complete steps unless --force
    if not args.force:
        steps_to_run = [
            s for s in steps_to_run
            if state.get(s["name"], {}).get("status") != "complete"
        ]

    print_banner()
    print(_c(f"  Config    : {args.config}", DIM))
    print(_c(f"  State file: {state_file}", DIM))
    print(_c(f"  Log dir   : {log_dir}", DIM))
    print()
    print_plan(steps_to_run, state)

    if not steps_to_run:
        print(_c("  Nothing to do — all selected steps are already complete.", GREEN))
        print(_c("  Use --force to re-run them.\n", DIM))
        return

    if args.dry_run:
        print(_c("  Dry-run mode — no commands will be executed.\n", YELLOW))

    # Credentials
    needs_auth = any(s["needs_casda_auth"] for s in steps_to_run)
    env = os.environ.copy()

    if needs_auth and not args.no_credentials and not args.dry_run:
        # Prefer existing env vars; only prompt if not already set
        if env.get("CASDA_USER") and env.get("CASDA_PASS"):
            print(_c("  Using CASDA credentials from environment.\n", DIM))
        else:
            username, password = prompt_casda_credentials()
            env["CASDA_USER"] = username
            env["CASDA_PASS"] = password

    # Run steps
    results  = []
    total_t0 = time.monotonic()

    for step in steps_to_run:
        print_step_header(step)
        t0 = time.monotonic()

        ok, duration = run_step(
            step=step,
            config=args.config,
            log_dir=log_dir,
            env=env,
            dry_run=args.dry_run,
        )
        duration = time.monotonic() - t0

        if not args.dry_run:
            if ok:
                mark_complete(state_file, state, step["name"], duration)
                print(_c(f"\n  ✓ Step {step['id']} completed in {duration:.0f}s.", GREEN))
            else:
                mark_failed(state_file, state, step["name"],
                            getattr(step, "_exit_code", -1), duration)
                print(_c(f"\n  ✗ Step {step['id']} FAILED after {duration:.0f}s.", RED))
                print(_c(f"    Stopping. Fix the issue and re-run — completed steps "
                          f"will be skipped automatically.", YELLOW))
                results.append({"id": step["id"], "label": step["label"],
                                 "ok": False, "duration_s": duration})
                print_summary(results, time.monotonic() - total_t0)
                sys.exit(1)

        results.append({"id": step["id"], "label": step["label"],
                         "ok": ok, "duration_s": duration})

    all_ok = print_summary(results, time.monotonic() - total_t0)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
