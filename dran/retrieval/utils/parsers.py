""" Copied from hloc https://github.com/cvg/Hierarchical-Localization """
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict


def parse_image_list(path, with_intrinsics=False):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if with_intrinsics:
                camera_model, width, height, *params = data
                params = np.array(params, float)
                info = (camera_model, int(width), int(height), params)
                images.append((name, info))
            else:
                images.append(name)

    assert len(images) > 0
    logging.info(f'Imported {len(images)} images from {path.name}')
    return images


def parse_image_lists(paths, with_intrinsics=False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics)
    return images


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))

def parse_image_lists_with_intrinsics(paths):
    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

        logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
        for data in raw_data:
            data = data.strip('\n').split(' ')
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            info = (camera_model, int(width), int(height), params)
            results.append((name, info))

    assert len(results) > 0
    return results