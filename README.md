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
├── all_SBs/             # per-tile cutouts (generated)
├── annotations/         # COCO JSON stubs (generated)
└── Allmask/             # reference mask PNGs
outputs/
└── CatV1/v1_without_angle/   # Gal-DINO predictions + checkpoint
```

---

## Quickstart (bare metal / SLURM cluster)

```bash
# 1. Clone
git clone https://github.com/<your-org>/RGCAT.git
cd RGCAT

# 2. Create conda environment
conda create -n rgcat python=3.10
conda activate rgcat
conda install -c pytorch pytorch torchvision pytorch-cuda=12.1

# 3. Install pipeline deps
pip install -r requirements.txt

# 4. Compile GalDINO CUDA deformable attention ops
cd galdino/models/dino/ops
python setup.py build install
cd ../../../../

# 5. Edit config
#    Set paths in config/pipeline.yaml to point at your data mount

# 6. Run over all tiles
python catalog_script.py --config config/pipeline.yaml
```

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
