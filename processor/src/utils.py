import os
import sys
import numpy as np

from time import time
from scipy import ndimage
from skimage import morphology, measure

from tiger.masks import BoundingBox, retain_largest_components
from tiger.patches import PatchExtractor3D, compute_crop


class Timer:
    def __init__(self):
        self.start_time = time()

    def elapsed(self):
        current_time = time()
        elapsed = current_time - self.start_time
        self.start_time = current_time
        return elapsed


class VertebralBodySegmentationRefiner:
    def __init__(self, skip_incomplete: bool = True):
        self.skip_incomplete = skip_incomplete
        self.patch_shape = (128, 128, 128)

    def _extract_patch(self, mask, center):
        extractor = PatchExtractor3D(mask)
        mask_patch = extractor.extract_cuboid(center, self.patch_shape)
        return mask_patch.astype('uint8')

    def refine(self, vertebra_mask: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
        # Prepare empty mask, add refined vertebral body segmentations
        refined_mask = np.zeros_like(body_mask)

        body_labels = np.unique(body_mask)
        for label in np.unique(vertebra_mask):
            if label <= 0:
                continue
            if self.skip_incomplete and label > 100:
                continue

            # Check what the label in the other mask is
            if label in body_labels:
                body_label = label
            elif label > 100 and label - 100 in body_labels:
                body_label = label - 100
            elif label < 100 and label + 100 in body_labels:
                body_label = label + 100
            else:
                continue

            # Limit operations to a smaller patch
            bb = BoundingBox(vertebra_mask == label)
            patch_center = np.round(bb.center).astype(int)
            vertebra_patch = self._extract_patch(vertebra_mask == label, patch_center)
            body_patch = self._extract_patch(body_mask == body_label, patch_center)

            # Subtract body from full vertebra
            pe = vertebra_patch - body_patch

            # Run opening to disconnect vertebral body and posterior elements
            pe = morphology.opening(pe, morphology.cube(width=3))

            # Retain largest connected component
            pe = retain_largest_components(pe, labels=[1])

            # Subtract this component from the full mask
            nb = vertebra_patch - pe

            # Run median filter to refine mask
            nb = ndimage.median_filter(nb, size=2)

            # Keep only voxels that are also in the original mask
            nb = nb * vertebra_patch

            # Retain largest connected component
            nb = retain_largest_components(nb, labels=[1])

            # Copy into new mask
            refined_patch, new_body_patch = extract_valid_patch_pairs(refined_mask, nb, patch_center)
            unclaimed_voxels = refined_patch == 0
            refined_patch[unclaimed_voxels] = new_body_patch[unclaimed_voxels] * body_label

        return refined_mask


def remove_low_intensity_surface_voxels(image, segmentation, intensity_threshold):
    if intensity_threshold > -1000:
        low_intensity_voxels = image < intensity_threshold
        low_intensity_voxels[:, :, 0] = False
        low_intensity_voxels[:, :, -1] = False

        for label in np.unique(segmentation):
            if label == 0:
                continue

            voxels = segmentation == label
            surface_voxels_removed = morphology.binary_erosion(voxels)
            surface_voxels = np.logical_and(voxels, surface_voxels_removed == 0)
            removable_surface_voxels = np.logical_and(
                surface_voxels, low_intensity_voxels
            )
            segmentation[removable_surface_voxels] = 0

    return segmentation


def remove_small_components(image, mask, min_size):
    m = mask > 0

    outside_fov = image <= -2000
    m[outside_fov] = True

    cmap = measure.label(m, background=0)
    cmap[outside_fov] = 0
    labels, sizes = np.unique(cmap[cmap > 0], return_counts=True)

    for label, size in zip(labels, sizes):
        if size <= min_size:
            mask[cmap == label] = 0

    mask[outside_fov] = 0
    return mask


def extract_valid_patch_pairs(mask, mask_patch, patch_center):
    """ Extracts two patches of equal size corresponding to the valid area of the supplied patch """
    if mask.ndim == 4:
        crops = [compute_crop(c, s, a) for c, s, a in zip(patch_center, mask_patch.shape[1:], mask.shape[1:])]
        patch1 = mask[:, crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
        patch2 = mask_patch[
            (slice(None),) + tuple(slice(c[2] if c[2] > 0 else None, -c[3] if c[3] > 0 else None) for c in crops)
        ]
    else:
        crops = [compute_crop(c, s, a) for c, s, a in zip(patch_center, mask_patch.shape, mask.shape)]
        patch1 = mask[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1]]
        patch2 = mask_patch[
            tuple(slice(c[2] if c[2] > 0 else None, -c[3] if c[3] > 0 else None) for c in crops)
        ]
    return patch1, patch2


supported_image_extensions = ("mha", "mhd", "nii.gz", "nii")


def find_image_file(directory, basename):
    for ext in supported_image_extensions:
        f = os.path.join(directory, "{}.{}".format(basename, ext))
        if os.path.exists(f):
            return f
    return None


def find_image_files(directory):
    for basename in os.listdir(directory):
        for ext in supported_image_extensions:
            if basename.endswith(".{}".format(ext)):
                filename = os.path.join(directory, basename)
                if os.path.isfile(filename):
                    yield filename, basename[:-(len(ext) + 1)]
                    break


def warn(msg):
    print(msg, file=sys.stderr, flush=True)
