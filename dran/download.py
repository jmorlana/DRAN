""" Copied from PixLoc https://psarlin.com/pixloc/ """
import argparse
import requests
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
import zipfile
import tarfile
from typing import Optional, List

from . import settings, logger
from .run_CMU import TEST_SLICES_CMU, TRAINING_SLICES_CMU


URLs = dict(
    logs='https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/',
    Aachen='https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/',
    RobotCar='https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/',
    CMU='https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/'
)


def download_from_url(url: str, save_path: Path, overwrite: bool = False,
                      exclude_files: Optional[List[str]] = None,
                      exclude_dirs: Optional[List[str]] = None):
    subpath = Path(urlsplit(url).path)
    num_parents = len(subpath.parents)
    exclude = ['index.html*']
    if exclude_files is not None:
        exclude += exclude_files
    cmd = [
        'wget', '-r', '-np', '-nH', '-q', '--show-progress',
        '-R', f'"{",".join(exclude)}"',
        '--cut-dirs', str(num_parents), url,
        '-P', str(save_path)
    ]
    if exclude_dirs is not None:
        path = Path(urlsplit(url).path)
        cmd += ['-X', ' '.join(str(path / d) for d in exclude_dirs)]
    if not overwrite:
        cmd += ['-nc']
    logger.info('Downloading %s.', url)
    subprocess.run(' '.join(cmd), check=True, shell=True)


def extract_zip(zippath: Path, extract_path: Optional[Path] = None,
                remove: bool = True):
    if extract_path is None:
        extract_path = zippath.parent
    logger.info('Extracting %s.', zippath)
    with zipfile.ZipFile(zippath, 'r') as z:
        z.extractall(extract_path)
    if remove:
        zippath.unlink()
    return zippath.parent / zippath.name


def extract_tar(tarpath: Path, extract_path: Optional[Path] = None,
                remove: bool = True):
    if extract_path is None:
        extract_path = tarpath.parent
    logger.info('Extracting %s.', tarpath)
    with tarfile.open(tarpath, 'r') as f:
        f.extractall(extract_path)
    if remove:
        tarpath.unlink()


def download_Aachen(do_dataset=True, do_outputs=True):
    if do_dataset:
        url = URLs['Aachen']
        out_path = settings.DATA_PATH / 'Aachen'
        out_path.mkdir(exist_ok=True, parents=True)
        logger.info('Downloading the Aachen Day-Night dataset...')
        download_from_url(url + 'queries/', out_path / 'queries/')
        download_from_url(url + 'images/', out_path / 'images/')
        extract_zip(out_path / 'images/database_and_query_images.zip')

    if do_outputs:
        url = URLs['logs'] + 'Aachen-Day-Night/'
        out_path = settings.LOC_PATH / 'Aachen'
        logger.info('Downloading logs for the Aachen Day-Night dataset...')
        download_from_url(url, out_path)


def download_CMU(do_dataset=True, do_outputs=True, do_training=True,
                 slices: Optional[List[int]] = None):
    if slices is None:
        slices = TEST_SLICES_CMU
        if do_training:
            slices = slices + TRAINING_SLICES_CMU
    
    if do_dataset:
        url = URLs['CMU']
        out_path = settings.DATA_PATH / 'CMU'
        out_path.mkdir(exist_ok=True, parents=True)
        logger.info('Downloading the Extended CMU Seasons dataset...')
        for i in slices:
            if (out_path / f'slice{i}').exists():
                continue
            download_from_url(url + f'slice{i}.tar', out_path)
            extract_tar(out_path / f'slice{i}.tar')

    if do_outputs:
        url = URLs['logs'] + 'Extended-CMU-Seasons/'
        out_dir = settings.LOC_PATH / 'CMU'
        logger.info('Downloading logs for the Extended CMU Seasons dataset...')
        for i in slices:
            if i in TEST_SLICES_CMU:
                out_path = out_dir / f'slice{i}/'
                if not out_path.exists():
                    download_from_url(url + f'slice{i}/', out_path)

    if do_training:
        tarfile = 'cmu_pixloc_training.tar.gz'
        url = URLs['logs'] + 'training/' + tarfile
        out_path = settings.DATA_PATH
        logger.info('Downloading the training data for CMU...')
        download_from_url(url, out_path)
        extract_tar(out_path / tarfile)


def download_RobotCar(do_dataset=True, do_outputs=True):
    if do_dataset:
        url = URLs['RobotCar']
        out_path = settings.DATA_PATH / 'RobotCar'
        out_path.mkdir(exist_ok=True, parents=True)
        logger.info('Downloading the RobotCar Seasons dataset...')
        download_from_url(url, out_path, exclude_dirs=['3D-models/'])
        for images in (out_path / 'images').glob('*.zip'):
            extract_zip(images)

    if do_outputs:
        url = URLs['logs'] + 'RobotCar-Seasons/'
        out_path = settings.LOC_PATH / 'RobotCar'
        logger.info('Downloading logs for the RobotCar Seasons dataset...')
        download_from_url(url, out_path)


def download_MegaDepth(do_training=True):
    logger.info(
        'Downloading the MegaDepth dataset is not automated. '
        'Please download the undistorted images and SfM models from '
        'https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset')  # noqa

    if do_training:
        tarfile = 'megadepth_pixloc_training.tar.gz'
        url = URLs['logs'] + 'training/' + tarfile
        out_path = settings.DATA_PATH
        logger.info('Downloading the training data for MegaDepth...')
        download_from_url(url, out_path)
        extract_tar(out_path / tarfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choices = ['Aachen', 'CMU', 'RobotCar', 'MegaDepth']
    parser.add_argument(
        '--select', default=choices, choices=choices, nargs='+',
        help='Data to download, default: all.')
    parser.add_argument(
        '--CMU_slices', choices=TEST_SLICES_CMU+TRAINING_SLICES_CMU, nargs='+',
        type=int, help='CMU slices to download, default: all.')
    parser.add_argument(
        '--training', action='store_true',
        help='Whether to download training data and dumps, default: false.')
    args = parser.parse_args()
    for choice in args.select:
        kws = {}
        if choice == 'CMU':
            kws['slices'] = args.CMU_slices
        if choice in ['CMU', 'MegaDepth']:
            kws['do_training'] = args.training
        vars()[f'download_{choice}'](**kws)
