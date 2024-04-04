""" Modified from hloc https://github.com/cvg/Hierarchical-Localization """
from pathlib import Path
from pprint import pformat
import argparse

from . import extract_features
from . import pairs_from_retrieval
from settings import DATA_PATH, LOC_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default=DATA_PATH / 'Aachen',
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default=LOC_PATH / 'Aachen', 
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=50,
                    help='Number of image pairs for loc, default: %(default)s')
args = parser.parse_args()

# Setup the paths
dataset = args.dataset
images = dataset / 'images/images_upright/'

outputs = args.outputs  # where everything will be saved
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
loc_pairs = outputs / f'release-dran-{args.num_loc}.txt'  # top-k retrieved by NetVLAD

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')

retrieval_conf = extract_features.confs['dran-sfm120k']

print(f'Dataset: {dataset}')
print(f'Will save everything to {outputs}')

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(
    global_descriptors, loc_pairs, args.num_loc,
    query_prefix='query', db_model=reference_sfm)

