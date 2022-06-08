import os
import numpy as np
import random
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask

################ Paths and other configs - Set these #################################
places_dir = '/scratch/lhz209/nood/data/places'
output_dir = '/scratch/lhz209/nood/data/places/ood1'

target_places = [
    ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
    ['ocean', 'lake/natural']]              # Water backgrounds

confounder_strength = 0.6 # Determines relative size of majority vs. minority groups
dataset_name = 'placesbg'
######################################################################################
random.seed(0)
### Assign places to train, val, and test set
place_ids_df = pd.read_csv(
    os.path.join(places_dir, 'categories_places365.txt'),
    sep=" ",
    header=None,
    names=['place_name', 'place_id'],
    index_col='place_id')


target_place_ids = []

place_filenames = []
for idx, subtype_target_places in enumerate(target_places):

    for target_place in subtype_target_places:
        # Read place filenames associated with target_place
        place_filenames += [
            f'/{target_place[0]}/{target_place}/{filename}' for filename in os.listdir(
                os.path.join(places_dir, 'data_large', target_place[0], target_place))
            if filename.endswith('.jpg')]

    random.shuffle(place_filenames)

# import sys; sys.exit()
### Write dataset to disk
output_subfolder = os.path.join(output_dir, dataset_name)
os.makedirs(output_subfolder, exist_ok=True)
os.makedirs(os.path.join(output_subfolder, 'land'), exist_ok=True)
os.makedirs(os.path.join(output_subfolder, 'water'), exist_ok=True)

for i in tqdm(range(len(place_filenames))):
    place_filepath = place_filenames[i][1:]
    place_category = place_filepath.split("/")[1]
    place_name = place_filepath.split("/")[2]

    place_path = os.path.join(places_dir, 'data_large', place_filepath)
    # if place_category in target_places[0]:
    if "forest" in place_category:
        output_path = os.path.join(output_subfolder, 'land', place_name)
    # elif place_category in target_places[1]:
    elif "ocean" in place_category or "lake" in place_category:
        output_path = os.path.join(output_subfolder, 'water', place_name)
    else:
        raise Exception(f'Category{place_category} not found')
    shutil.copyfile(place_path, output_path)
