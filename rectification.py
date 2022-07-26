import os
import cv2
import argparse

from typing import Generic, TypeVar, Any, List, Optional, Tuple

import numpy as np

try:
    from typing import Literal  # type: ignore
except ImportError:
    from typing_extensions import Literal

from rectification_utils.loop_zhang import stereo_rectify_uncalibrated as stereo_rectify_uncalibrated_lz

__all__ = ["Array"]

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    pass


__all__ = ["match_features", "find_fundamental_matrix", "draw_epi_lines", "estimate_epipoles", "skew", "normalize"]


DETECTOR_NORMS_DICT = {
    "SIFT": (cv2.SIFT_create(), cv2.NORM_L2),
    "ORB": (cv2.ORB_create(), cv2.NORM_HAMMING),
    "AKAZE": (cv2.AKAZE_create(), cv2.NORM_HAMMING),
    "BRISK": (cv2.BRISK_create(), cv2.NORM_HAMMING),
}
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6


def _init_detector_matcher(detector_name: str) -> Tuple[cv2.Feature2D, cv2.DescriptorMatcher]:
    try:
        detector, norm = DETECTOR_NORMS_DICT[detector_name]
    except KeyError:
        detector, norm = DETECTOR_NORMS_DICT["ORB"]

    flann_params = (
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        if norm == cv2.NORM_L2
        else dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    )
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    return detector, matcher



def match_features(
    img1: Array[Tuple[int, int], np.uint8],
    img2: Array[Tuple[int, int], np.uint8],
    detector_name: str = "ORB",
    ratio: float = 0.6,
) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
    assert img1.ndim == 2 and img1.dtype == np.uint8, "img1 is invalid"
    assert img2.ndim == 2 and img2.dtype == np.uint8, "img2 is invalid"

    keypoint_detector, keypoint_matcher = _init_detector_matcher(detector_name)

    kps1, des1 = keypoint_detector.detectAndCompute(img1, None)
    kps2, des2 = keypoint_detector.detectAndCompute(img2, None)
    matches = keypoint_matcher.knnMatch(des1, des2, k=2)

    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < ratio * m[1].distance]

    return kps1, kps2, matches


def find_fundamental_matrix(
    img1: Array[Tuple[int, int], np.uint8],
    img2: Array[Tuple[int, int], np.uint8],
    detector_name: str = "ORB",
    ratio: float = 0.6,
) -> Tuple[
    Optional[Array[Tuple[Literal[3], Literal[3]], np.float64]],
    Array[Tuple[int, Literal[2]], np.float64],
    Array[Tuple[int, Literal[2]], np.float64],
]:
    all_kps1, all_kps2, matches = match_features(img1, img2, detector_name, ratio)
    kps1 = np.asarray([all_kps1[m.queryIdx].pt for m in matches])
    kps2 = np.asarray([all_kps2[m.trainIdx].pt for m in matches])

    num_keypoints = len(matches)
    if num_keypoints < 7:
        return None, kps1, kps2

    flag = cv2.FM_7POINT if num_keypoints == 7 else cv2.FM_8POINT
    F, mask = cv2.findFundamentalMat(kps1, kps2, flag)

    # get inlier keypoints
    kps1 = kps1[mask.ravel() == 1]
    kps2 = kps2[mask.ravel() == 1]

    return F, kps1, kps2

def store_images(left, right, name):
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(here, "rectified_output")):
        os.mkdir(os.path.join(here, "rectified_output"))
    if not os.path.isdir(os.path.join(here, "rectified_output", "{}".format(name))):
        os.mkdir(os.path.join(here, "rectified_output", "{}".format(name)))
    cv2.imwrite(os.path.join(here, "rectified_output", "{}".format(name), "{}L.png".format(name)), left)
    cv2.imwrite(os.path.join(here, "rectified_output", "{}".format(name), "{}R.png".format(name)), right)
    TXT_PATH = os.path.join(here, "rectified_output", "textlist.txt")


def rectify(path1, path2):
    left_image = cv2.imread(path1)
    right_image = cv2.imread(path2)
    left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    F, kps1, kps2 = find_fundamental_matrix(left_image_gray, right_image_gray, "ORB")

    img_size = (left_image.shape[1], left_image.shape[0])
    H1_lz, H2_lz = stereo_rectify_uncalibrated_lz(F, img_size)

    left_rectified_lz = cv2.warpPerspective(left_image, H1_lz, img_size)
    right_rectified_lz = cv2.warpPerspective(right_image, H2_lz, img_size)
    name = ".".join(path1.split('/')[-1].split(".")[:-1])
    store_images(left_rectified_lz, right_rectified_lz, name)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Rectification tool of uncalibrated stereo image')

    parser.add_argument('--imgpath1', required = True, help='What is the path of left image?')
    parser.add_argument('--imgpath2', required = True, help='What is the path of right image?')

    args = parser.parse_args()

    rectify(args.imgpath1, args.imgpath2)