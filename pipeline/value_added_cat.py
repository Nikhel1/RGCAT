import pandas as pd
import numpy as np
import json, os
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io.votable import parse
from astropy.io import fits
import glob
import argparse

input_dir = "./outputs_gal/CatV1/v1_without_angle/"
input_dir_wise = "./CatalogNET_V1/EMUcatalog_main/catalogs_wise/"
input_dir_desi = "./CatalogNET_V1/EMUcatalog_main/catalogs_desi/"

def get_args_parser():
    parser = argparse.ArgumentParser('Value added Cats', add_help=False)
    parser.add_argument('--SBid', type=str, default='12346')
    return parser
    
parser = argparse.ArgumentParser('Value added Cats', parents=[get_args_parser()])
args = parser.parse_args()
SBid = args.SBid

tab = pd.read_csv(f'{input_dir}{SBid}/cons_catalogs/consolidated_catalog_lastv1.csv')

"""
print(tab['type'].unique())
print(len(tab))
print(len(np.where(tab['type']=='S')[0]))
print(len(np.where(tab['type']=='FR-II')[0]))
print(len(np.where(tab['type']=='FR-I')[0]))
print(len(np.where(tab['type']=='FR-X')[0]))
print(len(np.where(tab['type']=='R')[0]))
print(len(np.where(tab['type']=='C')[0]))
print(len(np.where(tab['type']!='S')[0]))
"""

tab2 = pd.DataFrame(columns=['EMU_source_name','EMU_ra_deg', 'EMU_dec_deg', #Gal-DINO
                             'EMU_flux_peak', 'EMU_flux_int', 'EMU_img_id', #'EMU_filename',
                             'EMU_radio_bbox', 'EMU_PA', 'EMU_ra_cen_bbox', 'EMU_dec_cen_bbox', 
                             'EMU_type', 'EMU_score', 'EMU_n_selavy_components',
                             'selavy_island_id', 'selavy_component_id', 'selavy_component_name', 'selavy_ra_hms_cont', # selavy
                             'selavy_dec_hms_cont', 'selavy_ra_deg_cont', 'selavy_dec_deg_cont', 'selavy_ra_err',
                             'selavy_dec_err', 'selavy_flux_peak', 'selavy_flux_peak_err','selavy_flux_int', 'selavy_flux_int_err',
                             'selavy_spectral_index',
                             'catwise_source_id', 'catwise_source_name', 'catwise_ra', 'catwise_dec', 'catwise_w1mag', # catwise
                             'catwise_w1sigm', 'catwise_w2mag', 'catwise_w2sigm', 'catwise_separation',
                             'ls_id', 'ls_ra', 'ls_dec', 'ls_type', 'ls_pmra', 'ls_pmdec', 'ls_parallax',
                             'ls_mag_g', 'ls_mag_r', 'ls_mag_i', 'ls_mag_z',
                             'ls_mag_w1', 'ls_mag_w2', 'ls_mag_w3', 'ls_mag_w4',
                             'ls_z_id', 'ls_z_phot_median', 'ls_z_phot_median_i',
                             'ls_z_phot_std_i', 'ls_z_spec'])

# Copy values from 'tab' to 'tab2'
tab2['EMU_source_name'] = tab['filename_cons']
tab2['EMU_ra_deg'] = tab['wise_ra_cons']
tab2['EMU_dec_deg'] = tab['wise_dec_cons']
tab2['EMU_flux_peak'] = tab['flux_peak_cons']
tab2['EMU_flux_int'] = tab['flux_int_cons']
tab2['EMU_img_id'] = tab['img_id_cons']
#tab2['EMU_filename'] = tab['filename_cons']
tab2['EMU_radio_bbox'] = tab['radio_bbox_cons']
tab2['EMU_PA'] = tab['PA']
tab2['EMU_ra_cen_bbox'] = tab['ra_deg_cons']
tab2['EMU_dec_cen_bbox'] = tab['dec_deg_cons']
tab2['EMU_type'] = tab['type']; tab2.replace({'EMU_type': {'C': 'Pec', 'S': 'C', 'FR-X': 'FR-x'}}, inplace=True)
tab2['EMU_score'] = tab['score']
tab2['EMU_n_selavy_components'] = tab['n_components']

tab2['selavy_island_id'] = tab['island_id']
tab2['selavy_component_id'] = tab['component_id']
tab2['selavy_component_name'] = tab['component_name']
tab2['selavy_ra_hms_cont'] = tab['ra_hms_cont']
tab2['selavy_dec_hms_cont'] = tab['dec_dms_cont']
tab2['selavy_ra_deg_cont'] = tab['ra_deg_cont']
tab2['selavy_dec_deg_cont'] = tab['dec_deg_cont']
tab2['selavy_ra_err'] = tab['ra_err']
tab2['selavy_dec_err'] = tab['dec_err']
tab2['selavy_flux_peak'] = tab['flux_peak']
tab2['selavy_flux_peak_err'] = tab['flux_peak_err']
tab2['selavy_flux_int'] = tab['flux_int']
tab2['selavy_flux_int_err'] = tab['flux_int_err']
tab2['selavy_spectral_index'] = tab['spectral_index']

tab2['catwise_source_id'] = None
tab2['catwise_source_name'] = None
tab2['catwise_ra'] = None
tab2['catwise_dec'] = None
tab2['catwise_w1mag'] = None
tab2['catwise_w1sigm'] = None
tab2['catwise_w2mag'] = None
tab2['catwise_w2sigm'] = None
tab2['catwise_separation'] = None

tab2['ls_id'] = None
tab2['ls_ra'] = None
tab2['ls_dec'] = None
tab2['ls_type'] = None
tab2['ls_pmra'] = None
tab2['ls_pmdec'] = None
tab2['ls_parallax'] = None
tab2['ls_mag_g'] = None
tab2['ls_mag_r'] = None
tab2['ls_mag_i'] = None
tab2['ls_mag_z'] = None
tab2['ls_mag_w1'] = None
tab2['ls_mag_w2'] = None
tab2['ls_mag_w3'] = None
tab2['ls_mag_w4'] = None

tab2['ls_z_id'] = None
tab2['ls_z_phot_median'] = None
tab2['ls_z_phot_median_i'] = None
tab2['ls_z_phot_std_i'] = None
tab2['ls_z_spec'] = None

# Assuming tab2['EMU_ra_deg'] and tab2['EMU_dec_deg'] contain RA and Dec values in degrees
ra_values = tab2['EMU_ra_deg'].values
dec_values = tab2['EMU_dec_deg'].values

# Create a SkyCoord object
coords = SkyCoord(ra=ra_values * u.degree, dec=dec_values * u.degree, frame='icrs')

# Get the names in the format JHHMMSS+DDMMSS
names = coords.to_string(style='hmsdms', precision=1, sep='', pad=True)

tab2['EMU_source_name'] = ['EMU J' + name[:-2].replace(' ', '') for name in names]

"""
# Stats for score>0.5 
id_scoreC0p5 = np.where((tab2['EMU_type']=='C') & (tab2['EMU_score']>0))[0]
print(len(id_scoreC0p5))
id_score0p5_com = np.where((tab2['EMU_type']=='C') & (tab2['EMU_score']>0.5) & (tab2['EMU_score']>0))[0]
print(len(id_score0p5_com), len(id_score0p5_com)/len(id_scoreC0p5))
id_scoreext0p5 = np.where((tab2['EMU_type']!='C') & (tab2['EMU_score']>0))[0]
print(len(id_scoreext0p5))
id_score0p5_ext = np.where((tab2['EMU_type']!='C')  & (tab2['EMU_score']>0.5) & (tab2['EMU_score']>0))[0]
print(len(id_score0p5_ext), len(id_score0p5_ext)/len(id_scoreext0p5))
"""

# Load the WISE catalog
tab_wise = pd.read_csv(f'{input_dir_wise}/{SBid}_catwise_catalog.csv')

# Assuming tab2['EMU_ra_deg'] and tab2['EMU_dec_deg'] contain RA and Dec values in degrees
ra_values_EMU = tab2['EMU_ra_deg'].values
dec_values_EMU = tab2['EMU_dec_deg'].values

# Create SkyCoord object for EMU coordinates
coords_EMU = SkyCoord(ra=ra_values_EMU * u.degree, dec=dec_values_EMU * u.degree, frame='icrs')

# Create SkyCoord object for WISE coordinates
coords_wise = SkyCoord(ra=tab_wise['ra'].values * u.degree, dec=tab_wise['dec'].values * u.degree, frame='icrs')

# Find the nearest neighbors within 10 arcseconds
idx, d2d, _ = coords_EMU.match_to_catalog_sky(coords_wise)
d2d_wise = d2d.arcsecond

"""
print('WISE sources within 3 arcsecond distance w.r.t Gal-DINO:', len(d2d.arcsecond[d2d.arcsecond<=3]))
print('WISE sources within 10 arcsecond distance w.r.t Gal-DINO:', len(d2d.arcsecond[d2d.arcsecond<10]))
"""

# Fill values in tab2 and set NaN for 30 arcseconds
idx_sep10 = np.where(d2d.arcsecond>10)[0]
tab2['catwise_source_id'] = tab_wise['source_id'].values[idx]; tab2.loc[idx_sep10, 'catwise_source_id'] = np.nan
tab2['catwise_source_name'] = tab_wise['source_name'].values[idx]; tab2.loc[idx_sep10, 'catwise_source_name'] = np.nan
tab2['catwise_ra'] = tab_wise['ra'].values[idx]; tab2.loc[idx_sep10, 'catwise_ra'] = np.nan
tab2['catwise_dec'] = tab_wise['dec'].values[idx]; tab2.loc[idx_sep10, 'catwise_dec'] = np.nan
tab2['catwise_w1mag'] = tab_wise['w1mag'].values[idx]; tab2.loc[idx_sep10, 'catwise_w1mag'] = np.nan
tab2['catwise_w1sigm'] = tab_wise['w1sigm'].values[idx]; tab2.loc[idx_sep10, 'catwise_w1sigm'] = np.nan
tab2['catwise_w2mag'] = tab_wise['w2mag'].values[idx]; tab2.loc[idx_sep10, 'catwise_w2mag'] = np.nan
tab2['catwise_w2sigm'] = tab_wise['w2sigm'].values[idx]; tab2.loc[idx_sep10, 'catwise_w2sigm'] = np.nan
tab2['catwise_separation'] = d2d.arcsecond; tab2.loc[idx_sep10, 'catwise_separation'] = np.nan

# Set NaN for >3 arcseconds for compact sources
idx_C_sep3 = np.where((tab2['catwise_separation']>3) & (tab2['EMU_type']=='C'))[0]
tab2.loc[idx_C_sep3, 'catwise_source_id'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_source_name'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_ra'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_dec'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_w1mag'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_w1sigm'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_w2mag'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_w2sigm'] = np.nan
tab2.loc[idx_C_sep3, 'catwise_separation'] = np.nan

# Get EMU-PS positions from CatWISE
id_notnan_all_wise = np.where(np.isnan(tab2['catwise_separation'])==False)[0]
tab2.loc[id_notnan_all_wise, 'EMU_ra_deg'] = tab2.loc[id_notnan_all_wise, 'catwise_ra']
tab2.loc[id_notnan_all_wise, 'EMU_dec_deg'] = tab2.loc[id_notnan_all_wise, 'catwise_dec']

# Load the DESI catalog
if os.path.exists(f'{input_dir_desi}/{SBid}_desi_dr10_catalog.csv'):
	tab_desi = pd.read_csv(f'{input_dir_desi}/{SBid}_desi_dr10_catalog.csv')
	if len(tab_desi)>1:
		# Assuming tab2['EMU_ra_deg'] and tab2['EMU_dec_deg'] contain RA and Dec values in degrees
		ra_values_EMU = tab2['EMU_ra_deg'].values
		dec_values_EMU = tab2['EMU_dec_deg'].values
		
		# Create SkyCoord object for EMU coordinates
		coords_EMU = SkyCoord(ra=ra_values_EMU * u.degree, dec=dec_values_EMU * u.degree, frame='icrs')
		
		# Create SkyCoord object for DESI coordinates
		coords_desi = SkyCoord(ra=tab_desi['ra'].values * u.degree, dec=tab_desi['dec'].values * u.degree, frame='icrs')
		
		# Find the nearest neighbors within 10 arcseconds
		idx, d2d, _ = coords_EMU.match_to_catalog_sky(coords_desi)
		d2d_desi = d2d.arcsecond
		
		# Fill values in tab2 and set NaN for 5 arcseconds
		idx_sep10 = np.where(d2d.arcsecond > 5)[0]
		tab2['ls_id'] = tab_desi['ls_id'].values[idx]; tab2.loc[idx_sep10, 'ls_id'] = np.nan
		tab2['ls_ra'] = tab_desi['ra'].values[idx]; tab2.loc[idx_sep10, 'ls_ra'] = np.nan
		tab2['ls_dec'] = tab_desi['dec'].values[idx]; tab2.loc[idx_sep10, 'ls_dec'] = np.nan
		tab2['ls_type'] = tab_desi['type'].values[idx]; tab2.loc[idx_sep10, 'ls_type'] = np.nan
		tab2['ls_pmra'] = tab_desi['pmra'].values[idx]; tab2.loc[idx_sep10, 'ls_pmra'] = np.nan
		tab2['ls_pmdec'] = tab_desi['pmdec'].values[idx]; tab2.loc[idx_sep10, 'ls_pmdec'] = np.nan
		tab2['ls_parallax'] = tab_desi['parallax'].values[idx]; tab2.loc[idx_sep10, 'ls_parallax'] = np.nan
		tab2['ls_mag_g'] = tab_desi['mag_g'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_g'] = np.nan
		tab2['ls_mag_r'] = tab_desi['mag_r'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_r'] = np.nan
		tab2['ls_mag_i'] = tab_desi['mag_i'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_i'] = np.nan
		tab2['ls_mag_z'] = tab_desi['mag_z'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_z'] = np.nan
		tab2['ls_mag_w1'] = tab_desi['mag_w1'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_w1'] = np.nan
		tab2['ls_mag_w2'] = tab_desi['mag_w2'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_w2'] = np.nan
		tab2['ls_mag_w3'] = tab_desi['mag_w3'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_w3'] = np.nan
		tab2['ls_mag_w4'] = tab_desi['mag_w4'].values[idx]; tab2.loc[idx_sep10, 'ls_mag_w4'] = np.nan
		
		# Set NaN for >3 arcseconds for compact sources
		idx_C_sep3 = np.where((tab2['catwise_separation'] > 1) & (tab2['EMU_type'] == 'C'))[0]
		tab2.loc[idx_C_sep3, 'ls_id'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_ra'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_dec'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_type'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_pmra'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_pmdec'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_parallax'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_g'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_r'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_i'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_z'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_w1'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_w2'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_w3'] = np.nan
		tab2.loc[idx_C_sep3, 'ls_mag_w4'] = np.nan
		
		# Load the DESI photoz catalog
		if os.path.exists(f'{input_dir_desi}/{SBid}_desi_dr10_z_catalog.csv'):
			tab_desi_z = pd.read_csv(f'{input_dir_desi}/{SBid}_desi_dr10_z_catalog.csv')
			if len(tab_desi_z)>1:
				# Merge tab2 with tab_desi_z on 'ls_id'
				merged = tab2.merge(
					tab_desi_z[['ls_id', 'z_phot_median', 'z_phot_median_i', 'z_phot_std_i', 'z_spec']],
					on='ls_id',
					how='left'
				)
				
				# Assign merged values back to tab2
				tab2['ls_z_id'] = merged['ls_id'].where(merged['z_phot_median'].notna(), None)
				tab2['ls_z_phot_median'] = merged['z_phot_median']
				tab2['ls_z_phot_median_i'] = merged['z_phot_median_i']
				tab2['ls_z_phot_std_i'] = merged['z_phot_std_i']
				tab2['ls_z_spec'] = merged['z_spec']
				
				# Create idx_all with index positions from tab_desi_z or None
				ls_id_to_index = {id_: idx for idx, id_ in enumerate(tab_desi_z['ls_id'])}
				idx_all = [ls_id_to_index.get(ls_id, None) for ls_id in tab2['ls_id']]
else:
	print(f'{input_dir_desi}/{SBid}_desi_dr10_catalog.csv does not exists.')

# All sources
tab2.to_csv(f"{input_dir}{SBid}/cons_catalogs/value_added_catwise_ls_all.csv", index=False)
data = pd.read_csv(f"{input_dir}{SBid}/cons_catalogs/value_added_catwise_ls_all.csv", usecols=['EMU_source_name', 'EMU_ra_deg', 'EMU_dec_deg', 'EMU_type', 'EMU_score'], low_memory=False)
table = Table.from_pandas(data)
table.write(f"{input_dir}{SBid}/cons_catalogs/value_added_catwise_ls_all.fits", format='fits', overwrite=True)

# All extended sources
which_sources = 'allExtended'
tab3 = tab2.iloc[list(tab2['EMU_type'] != 'C')]; tab3 = tab3.reset_index(drop=True)
tab3.drop('catwise_separation', axis=1, inplace=True)
tab3.to_csv(f"{input_dir}{SBid}/cons_catalogs/value_added_catwise_ls_{which_sources}.csv", index=False)
data1 = pd.read_csv(f"{input_dir}{SBid}/cons_catalogs/value_added_catwise_ls_{which_sources}.csv", usecols=['EMU_source_name', 'EMU_ra_deg', 'EMU_dec_deg', 'EMU_type', 'EMU_score'], low_memory=False)
table = Table.from_pandas(data1)
table.write(f"{input_dir}{SBid}/cons_catalogs/value_added_catwise_ls_{which_sources}.fits", format='fits', overwrite=True)
