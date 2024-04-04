""" Modified from hloc https://github.com/cvg/Hierarchical-Localization """
import argparse
import torch
from pathlib import Path
import h5py
import logging
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image

from . import extractors
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'dran-sfm120k': {
        'output': 'sfm120k-feats-dran',
        'model': {
            'name': 'dran',
            'dataset': 'sfm120k',
            'learned_whiten': True,
        },
        'preprocessing': {
            'resize_max': 1024,
        },
    },
    'dran-msls': {
        'output': 'msls-feats-dran-release',
        'model': {
            'name': 'dran',
            'dataset': 'msls',
            'learned_whiten': False,
        },
        'preprocessing': {
            'resize_max': 1024,
        },
    },
}


def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_'):].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_linear',  # switch to pil_linear for accuracy
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        if paths is None:
            paths = []
            for g in conf.globs:
                paths += list(Path(root).glob('**/'+g))
            if len(paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            paths = sorted(list(set(paths)))
            self.names = [i.relative_to(root).as_posix() for i in paths]
            logging.info(f'Found {len(self.names)} images in root {root}.')
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p
                              for p in paths]
            else:
                raise ValueError(f'Unknown format for path argument {paths}.')

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(
                        f'Image {name} does not exists in root: {root}.')

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(size) > self.conf.resize_max):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': name,
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


@torch.no_grad()
def main(conf, image_dir, export_dir=None, as_half=False,
         image_list=None, feature_path=None):
    logging.info('Extracting local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    loader = ImageDataset(image_dir, conf['preprocessing'], image_list)
    loader = torch.utils.data.DataLoader(loader, num_workers=1)

    if feature_path is None:
        feature_path = Path(export_dir, conf['output']+'.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(list_h5_names(feature_path)
                     if feature_path.exists() else ())
    if set(loader.dataset.names).issubset(set(skip_names)):
        logging.info('Skipping the extraction.')
        return feature_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    for data in tqdm(loader):
        name = data['name'][0]  # remove batch dimension
        if name in skip_names:
            continue

        pred = model(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), 'a') as fd:
            try:
                grp = fd.create_group(name)
                #print(pred)
                for k, v in pred.items():
                    # print(k, v)
                    # print(v.shape)
                    # if k == 'global_descriptor':
                    #     grp.create_dataset(k, data=v)
                    grp.create_dataset(k, data=v)
            except OSError as error:
                if 'No space left on device' in error.args[0]:
                    logging.error(
                        'Out of disk space: storing features on disk can take '
                        'significant space, did you enable the as_half flag?')
                    del grp, fd[name]
                raise error

        del pred

    logging.info('Finished exporting features.')
    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    parser.add_argument('--as_half', action='store_true')
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--feature_path', type=Path)
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
