""" Modified from PixLoc https://psarlin.com/pixloc/ """
import pickle

from . import set_logging_debug
from .localization import RetrievalLocalizer, PoseLocalizer
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results


default_paths = Paths(
    query_images='images/images_upright/',
    reference_images='images/images_upright/',
    reference_sfm='sfm_superpoint+superglue/',
    query_list='*_time_queries_with_intrinsics.txt',
    retrieval_pairs='release-dran-50.txt',
    results='release_dran_aachen.txt',
)

experiment = 'dran_sfm120k_megadepth'

default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {
             'conf':{
                'training': False,
                'load_retrieval': False,
                'rank': True,
                'light': False,
            },
        },
        'optimizer': {
            'num_iters': 150, 
            'pad': 1,
        },
        'refinement': {
            'num_dbs': 5,
            'multiscale': [1],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': False,
            'do_pose_approximation': False,
            'do_rerank_PnP': True,
            's2dhm_factor': 0.12,
            'only_pnp': False,
            'light': False,
        },
    },
    'from_poses': {
        'experiment': experiment,
        'features': {'preprocessing': {'resize': 1600}},
        'optimizer': {
            'num_iters': 50,
            'pad': 1,
        },
        'refinement': {
            'num_dbs': 5,
            'min_points_opt': 100,
            'point_selection': 'inliers',
            'normalize_descriptors': True,
            'average_observations': True,
            'layer_indices': [0, 1],
        },
    },
}


def main():
    parser = create_argparser('Aachen')
    args = parser.parse_intermixed_args()
    set_logging_debug(args.verbose)
    paths = parse_paths(args, default_paths)
    conf = parse_conf(args, default_confs)

    print(paths)

    if args.from_poses:
        localizer = PoseLocalizer(paths, conf)
    else:
        localizer = RetrievalLocalizer(paths, conf)
    poses, logs = localizer.run_batched(skip=args.skip)

    write_pose_results(poses, paths.results)
    with open(f'{paths.results}_logs.pkl', 'wb') as f:
        pickle.dump(logs, f)


if __name__ == '__main__':
    main()
