""" Modified from hloc https://github.com/cvg/Hierarchical-Localization """
from pathlib import Path
import argparse

from . import extract_features
from . import robotcar_pairs_from_retrieval
from settings import DATA_PATH, LOC_PATH


CONDITIONS = ['dawn', 'dusk', 'night', 'night-rain', 'overcast-summer',
              'overcast-winter', 'rain', 'snow', 'sun']


def generate_query_list(dataset, image_dir, path):
    h, w = 1024, 1024
    intrinsics_filename = 'intrinsics/{}_intrinsics.txt'
    cameras = {}
    for side in ['left', 'right', 'rear']:
        with open(dataset / intrinsics_filename.format(side), 'r') as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ['SIMPLE_RADIAL', w, h, fx, cx, cy, 0.0]
            cameras[side] = [str(p) for p in params]

    queries = sorted(image_dir.glob('**/*.jpg'))
    queries = [str(q.relative_to(image_dir.parents[0])) for q in queries]

    out = [[q] + cameras[Path(q).parent.name] for q in queries]
    with open(path, 'w') as f:
        f.write('\n'.join(map(' '.join, out)))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=Path, default=DATA_PATH / 'RobotCar', #datasets/robotcar
                    help='Path to the dataset, default: %(default)s')
parser.add_argument('--outputs', type=Path, default=LOC_PATH / 'RobotCar',
                    help='Path to the output directory, default: %(default)s')
parser.add_argument('--num_covis', type=int, default=20,
                    help='Number of image pairs for SfM, default: %(default)s')
parser.add_argument('--num_loc', type=int, default=20,
                    help='Number of image pairs for loc, default: %(default)s')
args = parser.parse_args()

# Setup the paths
dataset = args.dataset
images = dataset / 'images/'

outputs = args.outputs  # where everything will be saved
outputs.mkdir(exist_ok=True, parents=True)
query_list = outputs / '{condition}_queries_with_intrinsics.txt'
reference_sfm = outputs / 'sfm_superpoint+superglue'
loc_pairs = outputs / f'release-dran-{args.num_loc}.txt'

reference_sfm = dataset / '3D-models'
location_dir = dataset / '3D-models/individual'

# pick one of the configurations for extraction and matching
retrieval_conf = extract_features.confs['dran-msls']

for condition in CONDITIONS:
    generate_query_list(
        dataset, images / condition,
        str(query_list).format(condition=condition))

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
# Per location and per camera
robotcar_pairs_from_retrieval.main(
    global_descriptors, loc_pairs, args.num_loc,
    query_prefix=CONDITIONS, db_prefix= 'overcast-reference',
    db_model=reference_sfm, per_camera=True, location_dir=location_dir)


