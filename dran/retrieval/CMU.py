""" Modified from hloc https://github.com/cvg/Hierarchical-Localization """
import logging
from pathlib import Path
import argparse
import torch
import numpy as np

from . import extract_features
from . import pairs_from_retrieval
from .cirtorch.utils.whiten import pcawhitenlearn
from settings import DATA_PATH, LOC_PATH

TEST_SLICES = [2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]

def generate_query_list(dataset, path, slice_):
    cameras = {}
    with open(dataset / 'intrinsics.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == '#' or line == '\n':
                continue
            data = line.split()
            cameras[data[0]] = data[1:]
    assert len(cameras) == 2

    queries = dataset / f'{slice_}/test-images-{slice_}.txt'
    with open(queries, 'r') as f:
        queries = [q.rstrip('\n') for q in f.readlines()]

    out = [[q] + cameras[q.split('_')[2]] for q in queries]
    with open(path, 'w') as f:
        f.write('\n'.join(map(' '.join, out)))


def run_slice(slice_, root, outputs, num_covis, num_loc, conf, m=None, P=None):
    dataset = root / slice_
    ref_images = dataset / 'database'
    query_images = dataset / 'query'
    sift_sfm = dataset / 'sparse'

    outputs = outputs / slice_
    outputs.mkdir(exist_ok=True, parents=True)
    query_list = dataset / 'queries_with_intrinsics.txt'
    loc_pairs = outputs / f'release-dran-{num_loc}.txt'
    ref_sfm = outputs / 'sfm_superpoint+superglue/model/'

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['dran-msls']

    generate_query_list(root, query_list, slice_)
    global_descriptors = extract_features.main(
        retrieval_conf, ref_images, outputs)
    global_descriptors = extract_features.main(
        retrieval_conf, query_images, outputs)
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_loc, m, P, 
        query_list=query_list, db_model=ref_sfm)

def learn_slice(slice_, root, outputs, num_covis, num_loc, conf):
    dataset = root / slice_
    ref_images = dataset / 'database'
    query_images = dataset / 'query'
    sift_sfm = dataset / 'sparse'

    outputs = outputs / slice_
    outputs.mkdir(exist_ok=True, parents=True)
    query_list = dataset / 'queries_with_intrinsics.txt'
    loc_pairs = outputs / f'gcl-msls_dran{num_loc}.txt'
    ref_sfm = outputs / 'sfm_superpoint+superglue/model/'

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['dran-msls']

    generate_query_list(root, query_list, slice_)
    global_descriptors = extract_features.main(
        retrieval_conf, ref_images, outputs)
    global_descriptors = extract_features.main(
        retrieval_conf, query_images, outputs)
    slice_db_desc = pairs_from_retrieval.get_globals(
        global_descriptors, loc_pairs, num_loc,
        query_list=query_list, db_model=ref_sfm)

    return slice_db_desc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slices', type=str, default='[2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21]',
                        help='a single number, an interval (e.g. 2-6), '
                        'or a Python-style list or int (e.g. [2, 3, 4]')
    parser.add_argument('--dataset', type=Path,
                        default=DATA_PATH / 'CMU',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path,
                        default=LOC_PATH / 'CMU',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_covis', type=int, default=20,
                        help='Number of image pairs for SfM, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of image pairs for loc, default: %(default)s')
    parser.add_argument('--conf', type=str, default='dran-msls',
                        help='conf')
    args = parser.parse_args()

    if args.slices == '*':
        slices = TEST_SLICES
 
    if '-' in args.slices:
        min_, max_ = args.slices.split('-')
        slices = list(range(int(min_), int(max_)+1))
    else:
        slices = eval(args.slices)
        if isinstance(slices, int):
            slices = [slices]

    global_feats = []
    for slice_ in slices:
        logging.info('Learn PCA on all db set on slice %s.', slice_)
        db_feats_slice = learn_slice(
            f'slice{slice_}', args.dataset, args.outputs,
            args.num_covis, args.num_loc, args.conf)
        global_feats.append(db_feats_slice)

    global_feats = torch.cat(global_feats)

    db_np = global_feats.cpu().numpy().T
    # Learn PCA-whitening from db frames
    m, P = pcawhitenlearn(db_np)
      
    for slice_ in slices:
        logging.info('Working on slice %s.', slice_)
        run_slice(
            f'slice{slice_}', args.dataset, args.outputs,
            args.num_covis, args.num_loc, args.conf, m, P)
