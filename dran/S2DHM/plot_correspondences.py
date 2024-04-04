import gin
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


@gin.configurable
def plot_image_retrieval(left_image_path: str,
                         right_image_path: str,
                         title: str,
                         export_folder: str,
                         export_filename: str):
    """Display (query, nearest-neighbor) pairs of images."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]
    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

@gin.configurable
def plot_correspondences(left_image_path: str,
                         right_image_path: str,
                         left_keypoints: List[cv2.KeyPoint],
                         right_keypoints: List[cv2.KeyPoint],
                         matches: np.ndarray,
                         title: str,
                         export_folder: str,
                         export_filename: str):
    """Display feature correspondences."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)

    target_size = 1024
    left_image = left_image.astype(np.float32)
    right_image = right_image.astype(np.float32)
    if (max(left_image.shape[:2]) > target_size):
        left_image, scale_resize = resize(left_image, target_size, max, 'linear')
    if (max(right_image.shape[:2]) > target_size):
        right_image, scale_resize = resize(right_image, target_size, max, 'linear')

    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)
    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    # Draw Lines and Points
    for m in matches:
        left = left_keypoints[m[0]].pt
        right = tuple(sum(x) for x in zip(
            right_keypoints[m[1]].pt, (left_image.shape[1], 0)))
        cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255), 2)
    
    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)
    
    #output = np.zeros((height, width, 3), dtype=np.uint8)
    #output = cv2.drawMatches(left_image, left_keypoints, right_image, right_keypoints, matches, None)

    # img_l = cv2.drawKeypoints(np.uint8(left_image), left_keypoints, None, color=(255,0,0))
    # img_r = cv2.drawKeypoints(np.uint8(right_image), right_keypoints, None, color=(0,255,0))

    # # output[0:left_image.shape[0], 0:left_image.shape[1]] = img_l
    # # output[0:right_image.shape[0], left_image.shape[1]:] = img_r[:]
    # output = img_r

    # fig = plt.figure(figsize=(16, 7), dpi=160)
    # plt.imshow(output)
    # plt.title(title)
    # plt.axis('off')
    # #fig.savefig(str(Path(export_folder, export_filename)))
    # plt.show()
    # plt.close(fig)

    # output = img_l
    # plt.imshow(output)
    # plt.title(title)
    # plt.axis('off')
    # #fig.savefig(str(Path(export_folder, export_filename)))
    # plt.show()
    # plt.close(fig)

@gin.configurable
def plot_detections(left_image_path: str,
                    right_image_path: str,
                    left_keypoints: np.ndarray,
                    right_keypoints: np.ndarray,
                    title: str,
                    export_folder: str,
                    export_filename: str):
    """Display Superpoint detections."""
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(cv2.imread(right_image_path), cv2.COLOR_BGR2RGB)
    Path(export_folder, export_filename).parent.mkdir(exist_ok=True, parents=True)

    height = max(left_image.shape[0], right_image.shape[0])
    width = left_image.shape[1] + right_image.shape[1]
    offset = left_image.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:left_image.shape[0], 0:left_image.shape[1]] = left_image
    output[0:right_image.shape[0], left_image.shape[1]:] = right_image[:]

    fig = plt.figure(figsize=(16, 7), dpi=160)
    plt.imshow(output)
    plt.scatter(
        left_keypoints.T[:, 0], left_keypoints.T[:, 1], c='red', s=5)
    plt.scatter(
        right_keypoints.T[:, 0] + offset, right_keypoints.T[:, 1], c='red', s=5)
    plt.title(title)
    plt.axis('off')
    fig.savefig(str(Path(export_folder, export_filename)))
    plt.close(fig)

def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # if (h_new % 2 != 0) or (w_new % 2 != 0):
        #     h_new = math.ceil(h_new / 2.) * 2
        #     w_new = math.ceil(w_new / 2.) * 2
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale