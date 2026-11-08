# casda/ — CASDA Data Download Scripts

This folder contains standalone scripts for downloading EMU data products from
[CASDA](https://data.csiro.au), downloading external photometry and redshift
catalogs, and generating WISE mosaics.
Run all scripts from the **repo root** (one level up from this folder).

---

## Scripts

| Script | Purpose |
|---|---|
| `catalog_download_casda.py` | Download selavy component catalogues from CASDA (XML → CSV) |
| `radio_image_download_casda.py` | Download EMU FITS continuum tiles from CASDA |
| `wise_mosaics.py` | Generate unWISE W1 mosaics for each EMU tile |
| `wise_mosaics.sh` | Shell wrapper for `wise_mosaics.py` (auto-retry on failure) |
| `download_catwise_allwise_catalogs.py` | Download CatWISE-2020 and AllWISE photometry from IRSA TAP |
| `download_desi_catalogs.py` | Download DESI DR10 photometry and photo-z from NOIRLab DataLab TAP |

---

## Dependencies

Install the full pipeline requirements from the repo root:

```bash
pip install -r requirements.txt
```

`wise_mosaics.py` additionally requires **MontagePy** and the underlying
**Montage** C library.  See the [Montage installation](#montage-installation)
section below.

---

## catalog_download_casda.py

Downloads EMU selavy component catalogues from the CASDA TAP service and
converts them from VOTable XML to CSV.

```bash
python casda/catalog_download_casda.py
```

- Queries `ivoa.obscore` for all `selavy-image.i.EMU*components.xml` files.
- Skips SBIDs already present in `data/catalogs/`.
- Prompts once for OPAL credentials; authenticates against CASDA before
  starting.
- Resolves each file's DataLink URL directly — no async staging job.
- On success writes `data/catalogs/AS201_Continuum_Component_Catalogue_<SBID>_<SBID>.csv`.

**Access notes:**  
Catalogues that return `#error: no permission` in the DataLink response are
under embargo.  The script logs these as `SKIP (embargoed)` and reports them
separately at the end.  Re-running will not help; project-level CASDA access
or waiting for public release is required.

---

## radio_image_download_casda.py

Downloads EMU continuum FITS tiles (`image.i.EMU_*.cont.taylor.0.restored.conv.fits`).

```bash
python casda/radio_image_download_casda.py
```

- Queries `ivoa.obscore` for all matching tiles.
- Skips files already present in `data/images/`.
- Uses DataLink direct download — same mechanism as the catalogue script.
- Supports resume on partial downloads (HTTP Range requests).
- Embargoed tiles are logged as `SKIP (embargoed)`.
- 403 Forbidden errors from Pawsey S3 indicate a transient bad pre-signed URL;
  re-running will fetch a fresh URL from DataLink.

---

## wise_mosaics.py / wise_mosaics.sh

Generates a single unWISE W1 mosaic for every EMU FITS tile in `data/images/`.

### Quick start

```bash
# Recommended: use the shell wrapper (handles auto-retry and crash cleanup)
bash casda/wise_mosaics.sh

# Or run the Python script directly
python casda/wise_mosaics.py --config config/pipeline.yaml
```

### What it does

For each EMU tile:
1. Reads the WCS footprint and pads it by `wise.footprint_pad_deg` degrees
   (default 1.5°).
2. Queries the unWISE tile index (`tiles.fits`) to find overlapping coadd IDs.
3. Downloads W1 cutout FITS files into a per-tile scratch directory under
   `data/images_wise/`.
4. Runs the Montage pipeline: reproject → background-model → coadd.
5. Saves the mosaic to `data/images_wise/<stem>_wise_mosaic.fits`.
6. Cleans up all scratch files, keeping only the final mosaic.

### Configuration (`config/pipeline.yaml`)

```yaml
wise:
  tiles_url: "http://unwise.me/data/allwise/unwise-coadds/fulldepth/tiles.fits"
  images_base_url: "http://unwise.me/data/neo3/unwise-coadds/fulldepth"
  max_tiles_per_footprint: 75   # skip tile if footprint overlaps more than this
  footprint_pad_deg: 1.5        # degrees of padding added to each side

skip:
  bad_tiles:              # SBIDs to skip entirely (WISE CRVAL mismatch etc.)
    - "70511"
    - ...
```

The `skip.bad_tiles` list is shared with the main pipeline — tiles listed
there are skipped for both inference and WISE mosaic generation.

### Shell wrapper (`wise_mosaics.sh`)

The bash wrapper should be preferred over calling `wise_mosaics.py` directly
because:
- Mosaicking can be interrupted by NFS load issues or walltime limits.
- Already-completed mosaics are skipped on each attempt (idempotent).
- Stale per-tile scratch directories left by a crash are cleaned up before
  each retry.
- Retries up to 5 times (`MAX_RETRIES` in the script) before giving up.

```bash
bash casda/wise_mosaics.sh [--config config/pipeline.yaml]
```

---

## download_catwise_allwise_catalogs.py

Downloads CatWISE-2020 and AllWISE photometry catalogs from
[IRSA TAP](https://irsa.ipac.caltech.edu/TAP) for every EMU tile.

```bash
python casda/download_catwise_allwise_catalogs.py [--config config/pipeline.yaml]
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/pipeline.yaml` | Path to pipeline config |
| `--buffer-deg` | `2.0` | Sky buffer around each tile footprint (degrees) |
| `--retries` | `5` | TAP query retries on failure |
| `--delay` | `10` | Seconds between retries |

For each SBID produces:
- `data/catalogs_wise/<SBID>_catwise_catalog.csv` — CatWISE-2020: positions and W1/W2 magnitudes
- `data/catalogs_wise/<SBID>_allwise_catalog.csv` — AllWISE: positions and W1–W4 photometry

Already-present files are skipped.  Uses a POINT-in-BOX ADQL query so it
handles RA wrap-around correctly.  Requires internet access to IRSA.

---

## download_desi_catalogs.py

Downloads DESI DR10 photometry and photometric-redshift catalogs from
[NOIRLab DataLab TAP](https://datalab.noirlab.edu/tap).

```bash
python casda/download_desi_catalogs.py [--config config/pipeline.yaml]
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/pipeline.yaml` | Path to pipeline config |
| `--buffer-deg` | `2.0` | Sky buffer around each tile footprint (degrees) |
| `--retries` | `5` | TAP query retries on failure |
| `--delay` | `10` | Seconds between retries |
| `--batch-size` | `10000` | `ls_id` batch size for photo-z queries |

For each SBID produces:
- `data/catalogs_desi/<SBID>_desi_dr10_catalog.csv` — positions, morphology, magnitudes (g/r/i/z/W1–W4), filtered to 15 < mag_r < 23 and mag_z < 22
- `data/catalogs_desi/<SBID>_desi_dr10_z_catalog.csv` — photo-z: `z_phot_median`, `z_phot_median_i`, `z_phot_std_i`, `z_spec`

The photo-z table is built by joining on `ls_id` values from the photometry
table, queried in batches of `--batch-size` to avoid TAP result-size limits.
Already-present files are skipped.

---

## Montage Installation

`wise_mosaics.py` uses **MontagePy** (Python bindings to the Montage C library).
MontagePy itself is a pure-Python wheel, but it shells out to a compiled
Montage installation.

### 1. Install the Montage C library

```bash
git clone https://github.com/Caltech-IPAC/Montage.git
cd Montage
make
```

#### Known build fixes (gcc 10, Linux, Python 3.10)

Several source-level patches are needed for a clean build:

**`Montage/Makefile.LINUX`** — add `-fcommon` to `CFLAGS` and remove `-lnsl`:

```makefile
CFLAGS = -g -I. -I../lib/include -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 \
         -std=c99 -fcommon
LIBS   = -L../lib -lwcs -lcfitsio -lcoord -lmtbl -lsvc -lwww \
         -lboundaries -lpixbounds -ltwoplane -lm
```

Remove all remaining `-lnsl` occurrences:

```bash
find * -name 'Makefile.LINUX' | xargs sed -i 's/-lnsl//g'
```

**`lib/src/coord/coord.h`** — line 4:
```c
extern int coord_debug;
```

**`lib/src/coord/convertCoordinates.c`** — add at line 5:
```c
int coord_debug;
```

**`util/MovingTarget/mMovingTarget.c`** — line 89:
```c
extern long nodeCount;
```

**`util/Search/mSearch.c`** — comment out line 75:
```c
/* long nodeCount; */
```

**`MontageLib/MakeHdr/montageMakeHdr.c`** — replace all 94 occurrences of
`input` with `hdr_input`.

**`MontageLib/Subimage/montageSubimage.c`**:
- Line 93: comment out `/* int haveBlank */`
- Line 168: add inside the struct braces: `int haveBlank;`

**`MontageLib/TANHdr/montageTANHdr.c`** — rename the local variable `epoch`
to `char_epoch` everywhere it appears (avoids a collision with a stdlib name):

| Original | Replace with |
|---|---|
| `char epoch[80];` | `char char_epoch[80];` |
| `strcpy(epoch, "");` | `strcpy(char_epoch, "");` |
| `strcpy(epoch, value);` | `strcpy(char_epoch, value);` |
| `printf("epoch = [%s]\n", epoch)` | `printf("epoch = [%s]\n", char_epoch)` |
| `sprintf(temp, "EPOCH = %s", epoch)` | `sprintf(temp, "EPOCH = %s", char_epoch)` |

After applying all patches:

```bash
make
```

### 2. Install MontagePy

```bash
pip install MontagePy==1.2.3
```

### 3. Verify

```python
from MontagePy.main import mAdd, mImgtbl
print("MontagePy OK")
```

---

## Citation

If you use these scripts, please cite:

```bibtex
@ARTICLE{2024PASA...41...27G,
  author  = {{Gupta}, Nikhel and {Norris}, Ray P. and others},
  title   = "{RG-CAT: Detection pipeline and catalogue of radio galaxies
              in the EMU pilot survey}",
  journal = {\pasa},
  year    = 2024,
  volume  = {41},
  eid     = {e027},
  doi     = {10.1017/pasa.2024.25},
}

@ARTICLE{2023PASA...40...64G,
  author  = {{Gupta}, Nikhel and others},
  title   = "{Radio Galaxy Zoo EMU: Towards a Semantic Radio Galaxy Morphology Taxonomy}",
  journal = {\pasa},
  year    = 2023,
  doi     = {10.1017/pasa.2023.64},
}
```
