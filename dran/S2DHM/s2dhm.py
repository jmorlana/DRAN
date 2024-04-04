"""Base Pose Predictor Classes.
"""
import gin
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from . import solve_pnp
from . import plot_correspondences


def _choose_best_prediction(predictions, query_image):
    """Pick the best prediction from the top-N nearest neighbors."""
    filename = output_converter(query_image)
    best_prediction = np.argmax([p.num_inliers for p in predictions])
    quaternion = predictions[best_prediction].quaternion
    matrix = predictions[best_prediction].matrix
    return [filename, *quaternion, *list(matrix[:3,3])], predictions[best_prediction]

def _plot_inliers(left_image_path, right_image_path, left_keypoints,
    right_keypoints, matches, title, export_filename):
    """Plot the inliers."""
    plot_correspondences.plot_correspondences(
        left_image_path=left_image_path,
        right_image_path=right_image_path,
        left_keypoints=left_keypoints,
        right_keypoints=right_keypoints,
        matches=matches,
        title=title,
        export_filename=export_filename,
        export_folder='/home/jmorlana/experiments_reuse/outputs/S2DHM/correspondences_RC/')

def save(self, predictions: List):
    """Export the predictions as a .txt file.

    Args:
        predictions: The list of predictions, where each line contains a
            [query name, quaternion, translation], as per the CVPR Visual
            Localization Challenge.
    """
    print('>> Saving predictions under {}'.format(self._output_filename))
    Path(self._output_filename).parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(np.array(predictions))
    df.to_csv(self._output_filename, sep=' ', header=None, index=None)

def output_converter(filename: str):
        """Convert an absolute filename the output prediction format."""
        return '/'.join(filename.split('/')[-2:])
