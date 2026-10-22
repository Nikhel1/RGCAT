#!/usr/bin/env python
"""
catalog_script.py
-----------------
Orchestrator for the full RGCAT pipeline.

For every EMU tile found in <emu_images_dir>/*.fits the script runs these
stages in order, skipping any that already have their sentinel output:

  1. pipeline/Download_allSources.py      — radio + WISE cutouts
  2. pipeline/Preprocess_allSources.py    — normalised PNGs + FITS
  3. pipeline/create_fake_json_All_fits.py — COCO annotation stubs
  4. Gal-DINO inference (SLURM job)        — per-source predictions
  5. pipeline/get_consol_cat.py            — consolidated FITS catalogue
  6. pipeline/value_added_cat.py           — CatWISE + DESI cross-match

All paths, skip-lists, and resource specs are read from config/pipeline.yaml
(or the file passed via --config).

Usage:
    # Process all tiles
    python catalog_script.py --config config/pipeline.yaml

    # Process a single tile
    python catalog_script.py --tile image.i.EMU_0044-46.SB50180....fits
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
import time
from string import Template

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("catalog_script")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RGCAT full pipeline orchestrator."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline.yaml",
        help="Path to pipeline YAML config (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--tile",
        type=str,
        default=None,
        help=(
            "Process a single tile filename (basename only). "
            "If omitted, all .fits files in emu_images_dir are processed."
        ),
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to write log output in addition to stdout.",
    )
    return parser


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load and return the YAML pipeline config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------
def run_command(command: str, sbid: str, stage: str) -> None:
    """Run *command* in a subprocess.

    Raises RuntimeError on non-zero exit so the caller can decide whether to
    skip the tile or abort entirely.
    """
    logger.info("[SB%s] %s", sbid, stage)
    proc = subprocess.run(command, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{stage} exited with code {proc.returncode} for SBid {sbid}"
        )


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

# Template for the Gal-DINO inference SLURM script.
# Uses Python string.Template so ${var} placeholders are substituted at
# runtime — no f-string interpolation confusion.
_SLURM_TEMPLATE = Template("""\
#!/bin/bash -l
#SBATCH --account=${account}
#SBATCH --job-name=galdino-${sbid}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${ntasks_per_node}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --gres=${gres}
#SBATCH --mail-type=${mail_type}
#SBATCH --mail-user=${mail_user}
#SBATCH --output=${log_dir}/galdino_${sbid}_%j.out

python -m torch.distributed.launch \\
    --nproc_per_node=${nproc_per_node} \\
    --master_port=${master_port} \\
    --use_env \\
    galdino/main.py \\
    -c ${dino_config} \\
    --eval \\
    --resume ${dino_checkpoint} \\
    --SBid ${sbid}
""")


def submit_slurm_job(script_content: str, sbid: str) -> str:
    """Write a temporary SLURM script, submit it, return the job ID string."""
    script_path = f"/tmp/galdino_{sbid}.sbatch"
    with open(script_path, "w") as f:
        f.write(script_content)
    result = subprocess.run(
        ["sbatch", script_path], capture_output=True, text=True
    )
    os.unlink(script_path)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for SBid {sbid}: {result.stderr.strip()}"
        )
    job_id = result.stdout.strip().split()[-1]
    logger.info("[SB%s] SLURM job submitted — job_id=%s", sbid, job_id)
    return job_id


def wait_for_job(job_id: str, poll_interval: int = 30) -> None:
    """Block until *job_id* is no longer in the SLURM queue.

    After the job disappears from squeue, checks sacct for the final state
    and raises RuntimeError for FAILED / CANCELLED / TIMEOUT jobs.
    """
    logger.info("Polling SLURM job %s (every %ds) ...", job_id, poll_interval)
    while True:
        result = subprocess.run(
            ["squeue", "--job", job_id, "--noheader"],
            capture_output=True, text=True,
        )
        if job_id not in result.stdout:
            break
        time.sleep(poll_interval)

    # Check final state via sacct
    sacct = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True,
    )
    states = [s.strip() for s in sacct.stdout.splitlines() if s.strip()]
    logger.info("Job %s finished with state(s): %s", job_id, states)
    bad_states = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    if any(s in bad_states for s in states):
        raise RuntimeError(
            f"SLURM job {job_id} ended with state(s): {states}"
        )


# ---------------------------------------------------------------------------
# Per-tile orchestration
# ---------------------------------------------------------------------------
def process_tile(
    tile_filename: str,
    cfg: dict,
    failed_tiles: list,
) -> None:
    """Run all pipeline stages for one EMU tile filename."""

    paths   = cfg["paths"]
    slurm   = cfg["slurm"]
    proc    = cfg["processing"]
    galdino = cfg["galdino"]

    # Extract SBid
    sbmatch = re.search(r"SB(\d+)", tile_filename)
    if not sbmatch:
        logger.warning("Cannot extract SBid from '%s' — skipping", tile_filename)
        return
    sbid = sbmatch.group(1)

    # --- Skip-list check (bad tiles) ----------------------------------------
    bad_tiles = [str(s) for s in cfg["skip"].get("bad_tiles", [])]
    if sbid in bad_tiles:
        logger.info("[SB%s] In bad_tiles skip list — skipping", sbid)
        return

    # --- WISE mosaic existence check ----------------------------------------
    emu_tile_path = os.path.join(paths["emu_images_dir"], tile_filename)
    wise_basename = tile_filename.replace(".fits", "_wise_mosaic.fits")
    wise_tile_path = os.path.join(paths["wise_images_dir"], wise_basename)
    if not os.path.exists(wise_tile_path):
        logger.info(
            "[SB%s] WISE mosaic not found (%s) — skipping",
            sbid, os.path.basename(wise_tile_path),
        )
        return

    logger.info("=" * 64)
    logger.info("[SB%s] Processing %s", sbid, tile_filename)

    pipeline_dir    = "pipeline"
    annotations_dir = paths["annotations_dir"]
    output_dir      = paths["output_dir"]
    all_sbs_dir     = paths["all_sbs_dir"]

    try:
        # ------------------------------------------------------------------ #
        # Stage 1 — Download cutouts                                          #
        # ------------------------------------------------------------------ #
        cutout_dir = os.path.join(all_sbs_dir, sbid, "All_Images_8arcmV1")
        if not os.path.exists(cutout_dir):
            n_workers = proc.get("n_workers")
            workers_arg = (
                f" --n_workers {n_workers}" if n_workers is not None else ""
            )
            backend = proc.get("parallel_backend", "process")
            run_command(
                (
                    f"python {pipeline_dir}/Download_allSources.py"
                    f" --Image_SBid {tile_filename}"
                    f" --emu_images_dir {paths['emu_images_dir']}"
                    f" --wise_images_dir {paths['wise_images_dir']}"
                    f" --emu_catalogs_dir {paths['emu_catalogs_dir']}"
                    f" --all_sbs_dir {all_sbs_dir}"
                    f" --cutout_size {proc['cutout_size']}"
                    f" --angular_size {proc['angular_size_arcmin']}"
                    f"{workers_arg}"
                    f" --parallel_backend {backend}"
                ),
                sbid, "Download_allSources",
            )
        else:
            logger.info("[SB%s] Cutouts exist — skipping download", sbid)

        # ------------------------------------------------------------------ #
        # Stage 2 — Preprocess                                                #
        # ------------------------------------------------------------------ #
        all_fits_dir = os.path.join(all_sbs_dir, sbid, "All_fits")
        if not os.path.exists(all_fits_dir):
            n_workers = proc.get("n_workers")
            workers_arg = (
                f" --n_workers {n_workers}" if n_workers is not None else ""
            )
            backend = proc.get("parallel_backend", "process")
            run_command(
                (
                    f"python {pipeline_dir}/Preprocess_allSources.py"
                    f" --Image_SBid {tile_filename}"
                    f" --emu_images_dir {paths['emu_images_dir']}"
                    f" --emu_catalogs_dir {paths['emu_catalogs_dir']}"
                    f" --all_sbs_dir {all_sbs_dir}"
                    f" --cutout_size {proc['cutout_size']}"
                    f" --epmask {proc['epmask']}"
                    f" --wpmask {proc['wpmask']}"
                    f" --sigma {proc['sigma']}"
                    f"{workers_arg}"
                    f" --parallel_backend {backend}"
                ),
                sbid, "Preprocess_allSources",
            )
        else:
            logger.info("[SB%s] All_fits exists — skipping preprocessing", sbid)

        # ------------------------------------------------------------------ #
        # Stage 3 — COCO annotation stub                                      #
        # ------------------------------------------------------------------ #
        ann_file = os.path.join(annotations_dir, f"instances_angle_{sbid}.json")
        if not os.path.exists(ann_file):
            run_command(
                (
                    f"python {pipeline_dir}/create_fake_json_All_fits.py"
                    f" --Image_SBid {tile_filename}"
                    f" --all_sbs_dir {all_sbs_dir}"
                    f" --annotations_dir {annotations_dir}"
                    f" --mask_path {paths['mask_dir']}"
                ),
                sbid, "create_fake_json_All_fits",
            )
        else:
            logger.info("[SB%s] Annotation JSON exists — skipping", sbid)

        # ------------------------------------------------------------------ #
        # Stage 4 — Gal-DINO inference (SLURM)                               #
        # ------------------------------------------------------------------ #
        res_file = os.path.join(output_dir, f"results0_all_{sbid}.json")
        if os.path.exists(ann_file) and not os.path.exists(res_file):
            slurm_script = _SLURM_TEMPLATE.substitute(
                account         = slurm["account"],
                sbid            = sbid,
                nodes           = slurm["inference"]["nodes"],
                ntasks_per_node = slurm["inference"]["ntasks_per_node"],
                mem             = slurm["inference"]["mem"],
                time            = slurm["inference"]["time"],
                cpus_per_task   = slurm["inference"]["cpus_per_task"],
                gres            = slurm["inference"]["gres"],
                mail_type       = slurm["mail_type"],
                mail_user       = slurm["mail_user"],
                log_dir         = paths["log_dir"],
                nproc_per_node  = galdino["nproc_per_node"],
                master_port     = galdino["master_port"],
                dino_config     = galdino["config"],
                dino_checkpoint = paths["dino_checkpoint"],
            )
            job_id = submit_slurm_job(slurm_script, sbid)
            wait_for_job(
                job_id,
                poll_interval=slurm.get("poll_interval_seconds", 30),
            )
        else:
            logger.info("[SB%s] Gal-DINO results exist — skipping inference", sbid)

        # ------------------------------------------------------------------ #
        # Stage 5 — Consolidated catalogue                                    #
        # ------------------------------------------------------------------ #
        cons_sentinel = os.path.join(
            output_dir, sbid, "cons_catalogs",
            "consolidated_catalog_last_allExtendedv1.fits",
        )
        if not os.path.exists(cons_sentinel):
            run_command(
                (
                    f"python {pipeline_dir}/get_consol_cat.py"
                    f" --Image_SBid {tile_filename}"
                    f" --emu_images_dir {paths['emu_images_dir']}"
                    f" --emu_catalogs_dir {paths['emu_catalogs_dir']}"
                    f" --all_sbs_dir {all_sbs_dir}"
                    f" --annotations_dir {annotations_dir}"
                    f" --output_dir {output_dir}"
                ),
                sbid, "get_consol_cat",
            )
        else:
            logger.info("[SB%s] Consolidated catalogue exists — skipping", sbid)

        # ------------------------------------------------------------------ #
        # Stage 6 — Value-added catalogue                                     #
        # ------------------------------------------------------------------ #
        skip_va = [str(s) for s in cfg["skip"].get("skip_value_added", [])]
        if sbid in skip_va:
            logger.info(
                "[SB%s] In skip_value_added list — skipping value-added", sbid
            )
            return

        va_sentinel = os.path.join(
            output_dir, sbid, "cons_catalogs",
            "value_added_catwise_ls_allExtended.fits",
        )
        if not os.path.exists(va_sentinel):
            run_command(
                (
                    f"python {pipeline_dir}/value_added_cat.py"
                    f" --SBid {sbid}"
                    f" --output_dir {output_dir}"
                    f" --wise_catalogs_dir {paths['wise_catalogs_dir']}"
                    f" --desi_catalogs_dir {paths['desi_catalogs_dir']}"
                ),
                sbid, "value_added_cat",
            )
        else:
            logger.info(
                "[SB%s] Value-added catalogue exists — skipping", sbid
            )

    except RuntimeError as exc:
        logger.error("[SB%s] Stage failed: %s", sbid, exc)
        failed_tiles.append(sbid)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    # Optional file log handler
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logging.getLogger().addHandler(fh)

    # Load config
    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    cfg = load_config(args.config)
    logger.info("Config loaded from %s", args.config)

    # Ensure log dir exists
    os.makedirs(cfg["paths"].get("log_dir", "logs"), exist_ok=True)

    # Gather tiles
    if args.tile:
        tiles = [args.tile]
    else:
        tile_pattern = os.path.join(cfg["paths"]["emu_images_dir"], "*.fits")
        tiles = [os.path.basename(p) for p in sorted(glob.glob(tile_pattern))]
        if not tiles:
            logger.error(
                "No .fits tiles found in %s", cfg["paths"]["emu_images_dir"]
            )
            sys.exit(1)
        logger.info("Found %d tiles to process", len(tiles))

    failed_tiles: list = []
    for tile_filename in tiles:
        process_tile(tile_filename, cfg, failed_tiles)

    if failed_tiles:
        logger.warning(
            "%d tile(s) failed: %s", len(failed_tiles), failed_tiles
        )
        sys.exit(1)
    else:
        logger.info("All tiles completed successfully.")


if __name__ == "__main__":
    main()
