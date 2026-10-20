import re
import glob
import subprocess
import os, time

# Directory path
directory = 'CatalogNET_V1/EMUcatalog_main/images'
out_directory = 'outputs_gal/CatV1/v1_without_angle'

# Get the list of files in the directory
emu_tiles = glob.glob(f'{directory}/*')

# Good tiles seen with eye
emu_bad_tiles = [f"{directory}/image.i.EMU_0021-09A.SB53313.cont.taylor.0.restored.conv.fits",
                 f"{directory}/image.i.EMU_0054-14.SB54773.cont.taylor.0.restored.conv.fits"]

emu_maybe_good_tiles = [f"{directory}/image.i.EMU_0052-37.SB46971.cont.taylor.0.restored.conv.fits",
                        f"{directory}/image.i.EMU_0118-37.SB46986.cont.taylor.0.restored.conv.fits"]

emu_good_tiles = [f"{directory}/image.i.EMU_0044-46.SB50180.cont.taylor.0.restored.conv.fits",
                  f"{directory}/image.i.EMU_0049-51.SB46951.cont.taylor.0.restored.conv.fits",
                  f"{directory}/image.i.EMU_0102-32.SB53293.cont.taylor.0.restored.conv.fits",
                  f"{directory}/image.i.EMU_0130-55.SB50048.cont.taylor.0.restored.conv.fits",
                  f"{directory}/image.i.EMU_0144-46.SB46966.cont.taylor.0.restored.conv.fits",
                  f"{directory}/image.i.EMU_0154-51.SB47136.cont.taylor.0.restored.conv.fits"]

# Function to run a command and check for errors
def run_command(command):
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Error running command: {command}")
        exit(1)

# Function to submit SLURM job and wait for it to finish
def submit_slurm_job(slurm_script):
    # Write the SLURM script to a temporary file
    with open('temp_slurm_script.sh', 'w') as f:
        f.write(slurm_script)

    # Submit the SLURM job and capture the job ID
    result = subprocess.run(['sbatch', 'temp_slurm_script.sh'], capture_output=True, text=True)
    os.remove('temp_slurm_script.sh')
    
    if result.returncode != 0:
        print(f"Error submitting SLURM job: {result.stderr}")
        exit(1)
    
    # Extract job ID from sbatch output
    job_id = result.stdout.strip().split()[-1]
    return job_id

# Function to wait for a SLURM job to finish
def wait_for_job(job_id):
    while True:
        result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
        if job_id not in result.stdout:
            break

for emu in emu_tiles: #emu_good_tiles:
    emu_tile = os.path.basename(emu)
    # Extract the SBid from the file name
    SBmatch = re.search(r'SB(\d+)', emu_tile)
    if SBmatch:
        SBid = SBmatch.group(1)
    else:
        print(f"SBid not found in {emu_tile}")
        continue

    # 73385, 73778, 71680, 70398, 71674 something wrong with WISE mosaic, CRVAL don't match to EMU, all_SBs/73778/All_Images_8arcmV1_wise is empty
    if SBid in ('70511', '70463', '72044', '72176', '66742', '70395', '70399', '67157', '73385', '73339', '69699', '73778', '70504', '70512', '71680', '70398', '71674'):   
        print(f'Skipping SB {SBid}')
        continue
    
    wise_tile = emu.replace('.fits', '_wise_mosaic.fits').replace('/images/', '/images_wise/')
    if not os.path.exists(wise_tile):
        print(f'Skipping SB {SBid} as infrared image is not available')
        continue

    # Check and run commands conditionally
    if not os.path.exists(f"CatalogNET_V1/all_SBs/{SBid}/All_Images_8arcmV1"):
        run_command(f"cd CatalogNET_V1 && python Download_allSources.py --Image_SBid {emu_tile}")
    else:
        print(f'CatalogNET_V1/all_SBs/{SBid}/All_Images_8arcmV1 directory exists, skipping downloading')

    if not os.path.exists(f"CatalogNET_V1/all_SBs/{SBid}/All_fits"):
        print('Running Image Preprocessing Now')
        run_command(f"cd CatalogNET_V1 && python Preprocess_allSources.py --Image_SBid {emu_tile}")
    else:
        print(f'CatalogNET_V1/all_SBs/{SBid}/All_fits directory exists, skipping preprocessing')

    if not os.path.exists(f"CatalogNET_V1/annotations/instances_angle_{SBid}.json"):
        run_command(f"cd CatalogNET_V1 && python create_fake_json_All_fits.py --Image_SBid {emu_tile}")
    else:
        print(f'CatalogNET_V1/annotations/instances_angle_{SBid}.json exists, skipping fake json generating.')

    # SLURM job script
    slurm_script = f"""#!/bin/bash -l
#SBATCH --account=OD-231147
#SBATCH --job-name=d-dino
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=160gb
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nikhel.gupta@csiro.au
source /home/gup039/.bashrc
conda activate ddino-v
module load cuda/12.1.0
python -m torch.distributed.launch --nproc_per_node=1 --master_port='29501' --use_env main.py -c config/DINO/DINO_4scale.py --eval --resume outputs_gal/CatV1/v1_without_angle/checkpoint.pth --SBid {SBid}
"""
    if os.path.exists(f"CatalogNET_V1/annotations/instances_angle_{SBid}.json") and not os.path.exists(f"{out_directory}/results0_all_{SBid}.json"):
        print('Running sbatch job now to get Gal-DINO predictions.')
        job_id = submit_slurm_job(slurm_script)
        wait_for_job(job_id)
        time.sleep(120)
    else:
        print('Gal-DINO predictions already exist.')

    # Run catalog pipeline
    if not os.path.exists(f"{out_directory}/{SBid}/cons_catalogs/consolidated_catalog_last_allExtendedv1.fits"):
        print('Running the catalog pipeline now...')
        run_command(f"python get_consol_cat.py --Image_SBid {emu_tile}")
    else:
        print(f'Catalogs for {SBid} already exists.')

    # Run value added catalog pipeline 54771
    if SBid in ('50428'):
        continue
    if not os.path.exists(f"{out_directory}/{SBid}/cons_catalogs/value_added_catwise_ls_allExtended.fits"):
        print(f'Generating value added catalogs now for {SBid}...')
        run_command(f"python value_added_cat.py --SBid {SBid}")
    else:
        print(f'Value added catalogs for {SBid} already exists, running for next tile.')
