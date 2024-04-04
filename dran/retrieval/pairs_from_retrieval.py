""" Modified from hloc https://github.com/cvg/Hierarchical-Localization """
import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch
import collections.abc as collections
import sys

dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary
from .utils.io import list_h5_names
from .cirtorch.utils.whiten import whitenapply

logging.basicConfig(level=logging.INFO)

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None and isinstance(names, (str, Path)):
        names = parse_image_lists(names)
    elif names is not None and isinstance(names, collections.Iterable):
        names = list(names)
    else:
        raise ValueError('Provide either prefixes of names, a list of '
                         'images, or a path to list file.')
    return names

def get_globals(descriptors, output, num_matched,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    logging.info('Getting global descriptors from a retrieval database.')

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())

    if db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
        if name2idx is None:
            with h5py.File(str(path), 'r') as fd:
                desc = [fd[n][key].__array__() for n in names]
        else:
            desc = []
            for n in names:
                with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                    desc.append(fd[n][key].__array__())

        return torch.from_numpy(np.stack(desc, 0)).to(device).float()

    db_desc = get_descriptors(db_names, db_descriptors, name2db).squeeze()
    
    return db_desc

def main(descriptors, output, num_matched, m=None, P=None,
         query_prefix=None, query_list=None,
         db_prefix=None, db_list=None, db_model=None, db_descriptors=None):
    logging.info('Extracting image pairs from a retrieval database.')

    # We handle multiple reference feature files.
    # We only assume that names are unique among them and map names to files.
    if db_descriptors is None:
        db_descriptors = descriptors
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    query_names_h5 = list_h5_names(descriptors)

    if db_model:
        images = read_images_binary(db_model / 'images.bin')
        db_names = [i.name for i in images.values()]
    else:
        db_names = parse_names(db_prefix, db_list, db_names_h5)
    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = parse_names(query_prefix, query_list, query_names_h5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
        if name2idx is None:
            with h5py.File(str(path), 'r') as fd:
                desc = [fd[n][key].__array__() for n in names]
        else:
            desc = []
            for n in names:
                with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                    desc.append(fd[n][key].__array__())

        return torch.from_numpy(np.stack(desc, 0)).to(device).float()

    db_desc = get_descriptors(db_names, db_descriptors, name2db)
    query_desc = get_descriptors(query_names, descriptors)

    # Apply PCA-whitening
    if m is not None and P is not None:
        db_desc = db_desc.squeeze()
        query_desc = query_desc.squeeze()
        whitening = True
    else:
        whitening = False

    if whitening:
        # load to np and dims DxN
        db_np = db_desc.cpu().numpy().T
        q_np = query_desc.cpu().numpy().T

        # Apply PCA-whitening to all
        db_lw  = whitenapply(db_np, m, P)
        q_lw = whitenapply(q_np, m, P)

        db_desc = torch.from_numpy(db_lw.T).to('cuda:0')
        query_desc = torch.from_numpy(q_lw.T).to('cuda:0')

    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    pairs = []
    for query, indices in zip(query_names, topk):
        for i in indices:
            pair = (query, db_names[i])
            pairs.append(pair)

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
