# RG-CAT — Radio Galaxy Catalogue Pipeline

An automated pipeline for generating a value-added catalogue of radio galaxies
from [EMU](https://www.atnf.csiro.au/projects/emu/) survey tiles.  
The pipeline uses [**Gal-DINO**](https://arxiv.org/abs/2403.14235), a heavily customised multi-modal detection transformer 
(originally based on [DINO](https://arxiv.org/abs/2203.03605))
fine-tuned on radio morphology classes:
FR-I, FR-II, FR-X, Relaxed (R), Complex (C), and Single (S).

---

## Pipeline overview

```
EMU FITS tile  ──►  Download_allSources.py   (radio + WISE cutouts)
                         │
                         ▼
               Preprocess_allSources.py       (log-scale normalise, RGB PNG)
                         │
                         ▼
           create_fake_json_All_fits.py        (COCO annotation stub)
                         │
                         ▼
              Gal-DINO inference (SLURM)       (per-source detections)
                         │
                         ▼
                get_consol_cat.py              (consolidated FITS catalogue)
                         │
                         ▼
                value_added_cat.py             (cross-match: CatWISE, LS, DESI)
```

`catalog_script.py` (repo root) orchestrates all stages, with idempotent
checks so already-completed stages are skipped on re-run.

Before running the pipeline, use the `casda/` scripts to download all required
data products:

```
casda/catalog_download_casda.py            → data/catalogs/         (selavy CSVs)
casda/radio_image_download_casda.py        → data/images/           (EMU FITS tiles)
casda/wise_mosaics.sh                      → data/images_wise/      (unWISE W1 mosaics)
casda/download_catwise_allwise_catalogs.py → data/catalogs_wise/    (CatWISE/AllWISE)
casda/download_desi_catalogs.py            → data/catalogs_desi/    (DESI DR10 + photo-z)
```

---

## Repository layout

```
RGCAT/
├── catalog_script.py          # orchestrator — run this
├── pipeline/
│   ├── Download_allSources.py
│   ├── Preprocess_allSources.py
│   ├── create_fake_json_All_fits.py
│   ├── get_consol_cat.py
│   ├── value_added_cat.py
│   └── src/                   # COCO annotation helpers
├── casda/                     # ← CASDA data download scripts (run before pipeline)
│   ├── run_all.py                             # orchestrator — run this first
│   ├── catalog_download_casda.py              # download selavy component catalogues
│   ├── radio_image_download_casda.py          # download EMU FITS tiles
│   ├── wise_mosaics.py                        # generate unWISE W1 mosaics
│   ├── wise_mosaics.sh                        # shell wrapper (auto-retry)
│   ├── download_catwise_allwise_catalogs.py   # CatWISE-2020 + AllWISE from IRSA
│   ├── download_desi_catalogs.py              # DESI DR10 photometry + photo-z
│   └── README.md                              # installation + usage for casda scripts
├── galdino/                   # Gal-DINO detection framework
│   ├── main.py
│   ├── engine.py
│   ├── models/
│   ├── datasets/
│   ├── util/
│   ├── config/DINO/
│   └── scripts/
├── splinecalibration/         # confidence score recalibration (external)
├── config/
│   ├── pipeline.yaml          # ← all paths, skip lists, resource specs
│   └── DINO/                  # GalDINO model config files
├── Dockerfile
├── requirements.txt
└── logs/
```

Data directories (not tracked in git, mount at runtime):
```
data/
├── images/              # EMU FITS tiles
├── images_wise/         # WISE mosaics
├── catalogs/            # selavy CSV catalogs
├── checkpoint.pth       # trained Gal-DINO model checkpoint
├── score_calib_files/   # score calibration files for spline recalibration
│   ├── instances_train2017.json           # (optional — see note below)
│   ├── instances_val2017.json
│   ├── results0_train.json                # Gal-DINO predictions on train set
│   └── results0_val.json                  # Gal-DINO predictions on val set
└── all_SBs/             # per-tile working area (generated; deleted after each tile succeeds)
    └── <SBid>/
        ├── All_Images_8arcmV1/           # radio cutouts
        ├── All_Images_8arcmV1_wise/      # WISE cutouts
        ├── All2017/                      # RGB PNGs (preprocessed)
        ├── All_fits/                     # preprocessed radio FITS
        └── instances_angle_<SBid>.json   # COCO annotation stub (generated)
outputs/
└── CatV1/               # Gal-DINO inference outputs (per-tile)
    └── <SBid>/
        ├── results0_all_<SBid>.json   # raw Gal-DINO detections
        ├── config_args_all.json       # full resolved config used at inference
        ├── config_args_raw.json       # raw CLI config args
        ├── config_cfg.py              # GalDINO model config snapshot
        ├── info.txt                   # inference run log / stats
        ├── cons_catalogs/             # consolidated & value-added catalogues
        └── region_files/              # DS9 region files per class
```

> `data/Allmask/` is no longer required. The per-tile COCO annotation stub
> is generated synthetically on the fly by `create_fake_json_All_fits.py`
> and written directly into `all_SBs/<SBid>/`.

> `data/score_calib_files/` is only needed when confidence score recalibration is
> enabled (the default). It holds the two fixed train/val annotation JSONs
> used to fit the spline recalibrator in `get_consol_cat.py`. If these files
> are not available, pass `--no_calibrate_score` to `get_consol_cat.py`.

> `data/all_SBs/<SBid>/` is deleted automatically in the background by
> `catalog_script.py` once all stages for that tile complete successfully.
> Failed tiles are left on disk to allow inspection and re-run.

---

## CASDA data download

Before running the pipeline you need three data products per EMU tile:
the radio FITS image, the selavy component catalogue, and the unWISE W1 mosaic.
The scripts in `casda/` handle all five download steps.  Run them from the
**repo root** using the orchestrator:

```bash
python casda/run_all.py
```

This prompts for OPAL credentials once, then runs all five steps in sequence,
skipping already-completed steps on re-run.  See the options below for
selective execution.

> Full installation and usage details are in [`casda/README.md`](casda/README.md).

### Orchestrator options

```bash
python casda/run_all.py --steps 4 5      # only external catalog steps (no auth)
python casda/run_all.py --from-step 3    # resume from WISE mosaics
python casda/run_all.py --force          # re-run everything
python casda/run_all.py --dry-run        # preview without executing
```

### Running steps individually

### 1. Download selavy catalogues

```bash
python casda/catalog_download_casda.py
```

Queries CASDA TAP for all `selavy-image.i.EMU*components.xml` files and
converts them to CSV in `data/catalogs/`.  Already-downloaded SBIDs are
skipped.  Tiles under embargo are logged as `SKIP (embargoed)`.

### 2. Download EMU radio tiles

```bash
python casda/radio_image_download_casda.py
```

Downloads `image.i.EMU_*.cont.taylor.0.restored.conv.fits` files to
`data/images/`.  Supports resume on partial downloads.  Embargoed tiles are
skipped with a clear message.

### Authentication

Both these scripts prompt for OPAL (CASDA) credentials once per run.
Create an account at <https://opal.atnf.csiro.au/> if you do not have one.

### 3. Generate WISE mosaics

Requires the **Montage** C library and **MontagePy**.  See
[`casda/README.md — Montage Installation`](casda/README.md#montage-installation)
for build instructions including the patches needed for gcc 10 / Linux.

```bash
bash casda/wise_mosaics.sh
```

Generates `data/images_wise/<stem>_wise_mosaic.fits` for each tile in
`data/images/`.  The shell wrapper auto-retries on failure and cleans up
scratch directories between attempts.  Tiles listed under `skip.bad_tiles` in
`config/pipeline.yaml` are skipped automatically.

### 4. Download CatWISE-2020 and AllWISE catalogs

```bash
python casda/download_catwise_allwise_catalogs.py
```

Queries [IRSA TAP](https://irsa.ipac.caltech.edu/TAP) for CatWISE-2020 and
AllWISE photometry covering each EMU tile footprint (plus a 2° buffer).
Saves `data/catalogs_wise/<SBID>_catwise_catalog.csv` and
`data/catalogs_wise/<SBID>_allwise_catalog.csv`.
No authentication required — IRSA is publicly accessible.

### 5. Download DESI DR10 photometry and photo-z

```bash
python casda/download_desi_catalogs.py
```

Queries [NOIRLab DataLab TAP](https://datalab.noirlab.edu/tap) for DESI DR10
photometry (`ls_dr10.tractor`) and photo-z (`ls_dr10.photo_z`) for each tile.
Photo-z is fetched in batches by `ls_id`.
Saves `data/catalogs_desi/<SBID>_desi_dr10_catalog.csv` and
`data/catalogs_desi/<SBID>_desi_dr10_z_catalog.csv`.
No authentication required — NOIRLab DataLab is publicly accessible.

---

## Quickstart (bare metal / SLURM cluster)

```bash
# 1. Clone
git clone https://github.com/<your-org>/RGCAT.git
cd RGCAT

# 2. Create conda environment
conda create -n rgcat python=3.10
conda activate rgcat

# 3. Install PyTorch for your hardware (auto-detects CUDA / ROCm / MPS / CPU)
python install_torch.py

# 4. Install pipeline deps
pip install -r requirements.txt

# 5. Compile GalDINO CUDA deformable attention ops
cd galdino/models/dino/ops
python setup.py build install
cd ../../../../

# 6. Edit config
#    Set paths in config/pipeline.yaml to point at your data mount

# 7. Run over all tiles
python catalog_script.py --config config/pipeline.yaml
```

---

## Environment setup by platform

`install_torch.py` handles torch installation automatically.
The table below shows what it selects and any extra steps needed per platform.

| Platform | Detected by | Torch variant | Extra steps |
|---|---|---|---|
| NVIDIA GPU (SLURM / cloud) | `nvcc` / `nvidia-smi` | CUDA wheel (cu121 / cu124 / …) | `module load cuda/<version>` on HPC |
| AMD GPU (ROCm) | `rocm-smi` / `rocminfo` | ROCm wheel (rocm6.x) | Install ROCm runtime first |
| Apple Silicon (M1/M2/M3) | `platform.machine == arm64` | Default PyPI wheel (MPS built-in) | None |
| CPU-only (laptop, CI) | fallback | CPU-only wheel | None |

### SLURM cluster (HPC)

Set these in `config/pipeline.yaml` under the `slurm` section:

```yaml
slurm:
  conda_env: "ddino-v"        # conda env to activate inside the job
  cuda_module: "cuda/12.1.0"  # module load line (set to "" to skip)
  inference:
    backend: "slurm"          # submit via sbatch
```

### Cloud instance / laptop (no SLURM)

```yaml
slurm:
  inference:
    backend: "local"          # run torchrun directly in the current shell
galdino:
  device: "auto"              # cuda → mps → cpu fallback
```

### AMD GPU (ROCm)

Install the ROCm runtime (6.x recommended), then:

```bash
python install_torch.py   # will select the ROCm wheel automatically
```

Set `galdino.device: "cuda"` in `pipeline.yaml` — PyTorch ROCm exposes its
devices under the CUDA API, so `torch.cuda.is_available()` returns `True`.

### CPU-only / Apple MPS

```bash
python install_torch.py   # selects cpu or MPS wheel automatically
```

Set `galdino.device: "auto"` (the default). Distributed inference is not
supported on MPS; `nproc_per_node` is automatically treated as 1.

---

## Docker

```bash
# Build
docker build -t rgcat:latest .

# Run a single tile (interactive, GPU required)
docker run --gpus all \
  -v /path/to/your/data:/workspace/data \
  -v /path/to/outputs:/workspace/outputs \
  rgcat:latest \
  python catalog_script.py \
    --config config/pipeline.yaml \
    --tile image.i.EMU_0044-46.SB50180.cont.taylor.0.restored.conv.fits
```

The model checkpoint must be present at the path specified in
`config/pipeline.yaml` (`paths.dino_checkpoint`).  
Either bake it into the image or mount it as a volume.

---

## Configuration

All environment-specific settings live in `config/pipeline.yaml`.  
Key sections:

| Section | Purpose |
|---|---|
| `paths` | Data directories, output dir, checkpoint path |
| `skip.bad_tiles` | SBids with broken WISE mosaics (skip entirely) |
| `skip.skip_value_added` | SBids to skip for value-added step only |
| `processing` | Cutout size, angular size, segmentation thresholds |
| `galdino` | Model config path, distributed launch params |
| `slurm` | Account, mail, resource specs for the inference job |

---

## Classes

| ID | Label | Description |
|---|---|---|
| 1 | FR-II | Edge-brightened DRAGN |
| 2 | FR-I | Edge-darkened / core-brightened |
| 3 | FR-X | Intermediate / undecided morphology |
| 4 | R | Relaxed / amorphous |
| 5 | C | Complex multi-component |
| 6 | S | Single compact source |

---

## Citation

If you use this pipeline or the Gal-DINO model, please cite:

```bibtex
@ARTICLE{2024PASA...41...27G,
       author = {{Gupta}, Nikhel and {Norris}, Ray P. and {Hayder}, Zeeshan and {Huynh}, Minh and {Petersson}, Lars and {Rosalind Wang}, X. and {Hopkins}, Andrew M. and {Andernach}, Heinz and {Gordon}, Yjan and {Riggi}, Simone and {Yew}, Miranda and {Crawford}, Evan J. and {Koribalski}, B{\"a}rbel and {Filipovi{\'c}}, Miroslav D. and {Kapi{\'n}ska}, Anna D. and {Shabala}, Stanislav and {Vernstrom}, Tessa and {Marvil}, Joshua R.},
        title = "{RG-CAT: Detection pipeline and catalogue of radio galaxies in the EMU pilot survey}",
      journal = {\pasa},
     keywords = {Galaxies: active, galaxies: peculiar, radio continuum: galaxies, Galaxy: evolution, methods: data analysis, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = 2024,
        month = apr,
       volume = {41},
          eid = {e027},
        pages = {e027},
          doi = {10.1017/pasa.2024.25},
archivePrefix = {arXiv},
       eprint = {2403.14235},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024PASA...41...27G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

The pipeline builds on:
- [Gal-DINO model (Gupta et al., 2024)](https://arxiv.org/abs/2312.00306) — multi-modal detection transformer backbone
- [EMU Survey](https://www.atnf.csiro.au/projects/emu/) — Evolutionary Map of the Universe

---

## License

MIT — see [LICENSE](LICENSE).
