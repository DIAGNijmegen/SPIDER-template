import numpy as np

from os import path, makedirs
from scipy import ndimage
from tempfile import TemporaryDirectory

from tiger.io import read_image, write_image
from tiger.patches import PatchExtractor3D
from tiger.resampling import resample_image, reorient_image, pad_or_crop_image, align_mask_with_image
from tiger.masks import retain_largest_components, BoundingBox, ConnectedComponents

import utils
import networks

# class SpineThresholdTestPipeline:
#     def __init__(self):
#

class SpineSegmentationPipeline:
    def __init__(self, training_set="up-mr-FinalSpineSegmentationDataset", *, tps=False, sigma=0, slow=False, device="cuda"):
        self.version = "1.9.0"
        self.patch_shape = (64, 192, 192)
        self.resolution = (2.0, 0.6, 0.6)

        self.max_vertebrae = 25
        self.max_steps = 15
        self.patch_center_convergence_threshold = 2
        self.min_size_init = 1000
        self.min_size_cont = 1000
        self.min_size_fragment = 500
        self.min_label_init = 1.5
        self.min_label_cont = 0.0
        self.min_label_accept = 0.75
        self.retry_margin = self.patch_shape[2] // 4
        self.slow = slow

        self.channels = self.infer_channels(training_set)
        self.network = networks.IterativeSegmentationNetworkDoubleMemoryState(n_filters=50, n_output_channels=3, n_input_channels=3, traversal_direction='up', device=device)

        self.training_sets = {
            "up-mr-FinalSpineSegmentationDataset": "/app/data/FinalSpineSegmentationWeights-epoch999999.pkl"
        }

        self.default_training_set = training_set
        self.active_training_set = None
        self.traversal_direction = None
        self.modality = None
        self.retain_posterior_components = True
        self.keep_only_largest_component = None
        self.default_sigma = sigma
        self.load_training_set(self.default_training_set)

    @staticmethod
    def infer_channels(name):
        return 2 if "multitask" in name else 1

    def load_training_set(self, name):
        if name == self.active_training_set:
            return

        if name not in self.training_sets:
            raise ValueError("Unknown training set")

        if self.infer_channels(name) != self.channels:
            raise ValueError("Cannot change number of filters")

        self.active_training_set = name
        self.network.restore(self.training_sets[name])

        n = name.split("-")
        self.traversal_direction = n[0]
        self.modality = n[1]

        # MR specifiy settings
        # if self.modality == "mr":
        #     self.min_size_cont = 100
        #     self.min_size_fragment = 100
        # else:
        #     self.min_size_cont = 1000
        #     self.min_size_fragment = 1000

    def probabilities_to_mask(self, probabilities, image_patch, completeness):
        # If the network thinks that this is a complete vertebra, we should keep only a single component
        # For incomplete vertebrae, the vertebra can be split into multiple components
        mask_patch = probabilities > 0.5
        if completeness:
            return retain_largest_components(mask_patch, labels=(True,), background=False, n=1)
        else:
            return utils.remove_small_components(image_patch, mask_patch, self.min_size_fragment)

    def retain_posterior_components_function(self, image, mask, fragment_size=500):
        # first remove small fragments
        mask = utils.remove_small_components(image, mask, min_size=fragment_size)
        # get connected components. Save the largest. Retain all components posterior of the largest component
        if np.count_nonzero(mask) > 0:
            mask_components = ConnectedComponents(mask)
            largest_component = mask_components.components[-1]
            largest_component_Y_mean = np.where(largest_component.mask)[1].mean()
            retain_components = []
            for component in mask_components.components:
                if np.where(component.mask)[1].mean() > largest_component_Y_mean:
                    retain_components.append(component)

            # Find lowest component in retain_components
            retain_components_Z_mean = [np.where(component.mask)[2].mean() for component in retain_components]
            if retain_components_Z_mean:
                lowest_component = retain_components_Z_mean.index(min(retain_components_Z_mean))
                # Only put lowest component in one mask
                retain_mask = largest_component.mask
                component = retain_components[lowest_component]
                retain_mask[component.mask] = True
            else:
                retain_mask = largest_component.mask
                retain_mask[component.mask] = True

            # Put all components that need to be retained in one mask
            # retain_mask = largest_component.mask
            # for component in retain_components:
            #     retain_mask[component.mask] = True

            mask = np.multiply(mask, retain_mask)

        return mask

    def put_patch_in_full_mask(self, segmentation, predicted_segmentation, patch_center, new_header, original_spacing, new_shape, original_shape, label, resample=True):
        # Create a temporary segmentation mask the same size as the resampled image
        segmentation_temp = np.zeros(new_shape, dtype='float32')

        # Put segmentation mask from the patch into the full sized output mask
        segmentation_patch, prediction_patch = utils.extract_valid_patch_pairs(segmentation_temp,
                                                                         predicted_segmentation,
                                                                         patch_center)
        segmentation_patch[segmentation_patch == 0] = prediction_patch[segmentation_patch == 0]

        # Resample mask to original image dimensions
        if resample:
            segmentation_temp = pad_or_crop_image(
                resample_image(segmentation_temp, new_header['spacing'], original_spacing, order=3),
                target_shape=original_shape
            )

        segmentation[segmentation_temp > 0.5] = label

        return segmentation

    def __call__(self, image_file, overlay_files, model=None, sigma=None, surface_erosion_threshold=-2000, screenshot_generator=None, screenshot_file=None):
        clock = utils.Timer()

        # Is the network in the status in which we need it? Otherwise, load different weights
        self.load_training_set(model if model is not None else self.default_training_set)
        print(' > Initializing: {:.1f} s'.format(clock.elapsed()))

        # Read the image
        original_image, header = read_image(image_file)
        print('image')
        original_shape = original_image.shape
        original_spacing = header['spacing']
        print(header.spacing)
        original_header = header.copy()
        if original_image.ndim != 3 or header.ndim != 3:
            shape = " x ".join(original_image.shape)
            raise RuntimeError(f"Image with shape {shape} is not 3D")
        print(' > Reading image: {:.1f} s'.format(clock.elapsed()))

        # Normalize the image
        if not header.has_default_direction():
            image, header = reorient_image(original_image, header)
            image_flipped = True
            print('flipped image')
            print(image.shape)
        else:
            image = np.copy(original_image)
            image_flipped = False

        # copy of the original header after it was potentially flipped because of a non identity direction matrix
        header_flipped = header.copy()
        shape_flipped = image.shape
        print(shape_flipped)

        if self.modality == "ct":
            # CT scans may have the range 0 to 4096 (intercept = 1000)
            if not np.any(image < 0):
                image -= 1000

        image = np.clip(image, -1000, 3096).astype("int16")

        if self.modality == "mr":
            # MR scans are normalized by clipping values above/below the 5% and 95% percentiles
            # and squeezing the remaining values into the range -1000 to 3096
            p5, p95 = np.percentile(image, (5, 95))
            image = (image.astype("float32") - p5.astype("float32")) / (
                p95 - p5
            ).astype("float32")
            image = np.clip(image * 4096 - 1000, -1000, 3096).astype('int16')

        print(' > Normalizing image: {:.1f} s'.format(clock.elapsed()))

        # Resample image to working resolution (client has to make sure it is properly oriented)
        image = resample_image(
            image, header["spacing"], self.resolution, order=0, prefilter=True
        )
        header["spacing"] = self.resolution
        new_shape = image.shape
        print('resampled image')
        print(image.shape)

        print(' > Resampling to standard resolution: {:.1f} s'.format(clock.elapsed()))

        # Smooth image?
        if sigma is None:
            sigma = self.default_sigma

        if sigma > 0.001:
            image = ndimage.filters.gaussian_filter(image, sigma)
            print(' > Smoothing with sigma = {:.3f}: {:.1f} s'.format(sigma, clock.elapsed()))
        else:
            print(' > Smoothing: disabled')

        # Create empty segmentation mask etc
        image_patch_extractor = PatchExtractor3D(
            image, pad_value=-2000, dtype="float32"
        )

        # Create an empty mask for the segmentation result. One same size as original image, one new resolution used for the memory state
        segmentation = np.zeros(shape_flipped, dtype='int16')
        segmentation_new_res = np.zeros_like(image, dtype='int16')
        segmentation_new_res_patches = PatchExtractor3D(segmentation_new_res, pad_value=0, dtype='float32')

        print('segmentation')
        print(segmentation.shape)
        print('segmentation_new_res')
        print(segmentation_new_res.shape)

        # Create an empty mask for the disc segmentation result
        segmentation_discs = np.zeros(shape_flipped, dtype='int16')
        segmentation_discs_new_res = np.zeros_like(image, dtype='int16')
        segmentation_discs_new_res_patches = PatchExtractor3D(segmentation_discs_new_res, pad_value=0, dtype='float32')

        # Create an empty mask for the SC segmentation result
        segmentation_sc = np.zeros(shape_flipped, dtype='int16')

        # Create an empty list for the predicted labels
        labels = dict()

        # Iterative over the image to find vertebrae
        scan_shape = np.asarray(image.shape, dtype=int)
        max_coords = scan_shape - 1

        if self.traversal_direction == "up":
            starting_coordinates = np.array((0, 0, 0), dtype=int)
        else:
            starting_coordinates = max_coords

        patch_center = np.copy(starting_coordinates)

        # # Use larger steps for large images
        # if self.slow or image.shape[2] < 500:
        #     traversal_factors = (2, 2, 2)
        # elif image.shape[2] < 1000:
        #     traversal_factors = (1, 1, 2)
        # else:
        #     traversal_factors = (1, 1, 1)

        # Traverse over the image, searching for the spine
        n_steps = 0
        n_steps_vertebra = 0
        n_detected_vertebrae = 0
        completely_visible_vertebrae = set()
        retried_with_cropped_image = False
        print(" > Traversing {}wards (image shape: {} x {} x {})".format(
            self.traversal_direction, image.shape[0], image.shape[1], image.shape[2]
        ))

        while n_detected_vertebrae < self.max_vertebrae:
            n_steps += 1

            # Extract image patch at current position and correspond part of the segmentation mask
            image_patch = image_patch_extractor.extract_cuboid(
                patch_center, self.patch_shape
            )
            segmentation_new_res_patch = segmentation_new_res_patches.extract_cuboid(patch_center, self.patch_shape)
            segmentation_discs_new_res_patch = segmentation_discs_new_res_patches.extract_cuboid(patch_center,
                                                                                                 self.patch_shape)

            # # Feed image patch through network
            # memory_state = segmentation_patch > 0
            # prediction = self.network.segment_and_classify(
            #     image_patch, memory_state, threshold_segmentation=False
            # )

            # Feed image patch through network
            memory_state = segmentation_new_res_patch > 0
            memory_state_discs = segmentation_discs_new_res_patch > 0
            prediction, classification, labeling = self.network.segment_and_classify(image_patch, memory_state=memory_state,
                                                                                memory_state_discs=memory_state_discs,
                                                                                threshold_segmentation=False) #todo misschien hier niet channel? zie oude code gecomment hierboven

            predicted_segmentation_vertebrae = prediction[0] > 0.5
            predicted_segmentation_discs = prediction[1] > 0.5

            predicted_segmentation_vertebrae_float = np.copy(prediction[0])
            predicted_segmentation_discs_float = np.copy(prediction[1])
            predicted_segmentation_sc_float = np.copy(prediction[2])

            predicted_completeness = classification[0][0] > 0.5
            predicted_label = labeling

            n_positive_voxels = np.count_nonzero(predicted_segmentation_vertebrae)
            n_detected_voxels = np.count_nonzero(predicted_segmentation_vertebrae & ~memory_state)

            if self.retain_posterior_components:
                predicted_segmentation_vertebrae = self.retain_posterior_components_function(image_patch, predicted_segmentation_vertebrae, self.min_size_fragment)
            elif self.keep_only_largest_component:
                predicted_segmentation_vertebrae = retain_largest_components(predicted_segmentation_vertebrae, labels=(True,), background=False, n=1, connectivity=int(3))
            else:
                predicted_segmentation_vertebrae = utils.remove_small_components(image_patch, predicted_segmentation_vertebrae, self.min_size_fragment)
            predicted_segmentation_vertebrae_float = np.multiply(predicted_segmentation_vertebrae_float, predicted_segmentation_vertebrae)

            detected_volume = np.count_nonzero(predicted_segmentation_vertebrae & ~memory_state)
            voxel_volume = self.resolution[0]*self.resolution[1]*self.resolution[2]
            detected_volume *= voxel_volume

            # Print out some stats
            print('Step {:<3}: {} -> {} voxels ({:.3f}), label: {}'.format(n_steps, patch_center, n_detected_voxels,
                                                                           classification[0][0], predicted_label))

            bb = BoundingBox(predicted_segmentation_vertebrae)
            # Check if a large enough fragment of vertebral bone was detected
            if (
                n_detected_voxels >= self.min_size_init
                and predicted_label >= self.min_label_init
            ) or (
                n_detected_voxels >= self.min_size_cont
                and predicted_label >= self.min_label_cont
            ) and not bb.empty:
                # Determine center of bounding box of the fragment -> center of next patch
                fragment_center = bb.center
                new_patch_center = np.clip(
                    patch_center + np.round(fragment_center).astype(int) - np.asarray(self.patch_shape) // 2, 0, max_coords)

                # Check if the patch moved or if the position has converged
                if n_steps_vertebra > 0 and (
                    n_steps_vertebra > self.max_steps or
                    max(abs(new_patch_center - patch_center)) <= self.patch_center_convergence_threshold
                ):
                    if predicted_label < self.min_label_accept:
                        print(' > false positive')
                        # break
                    else:
                        # Put segmentation mask from the patch into the full sized output mass
                        label = n_detected_vertebrae + 1
                        print(' >> vertebra {} detected (completely visible: {})'.format(label,
                                                                                         'yes' if predicted_completeness else 'no'))

                        segmentation_new_res_patch, prediction_patch = utils.extract_valid_patch_pairs(segmentation_new_res,
                                                                                                 predicted_segmentation_vertebrae,
                                                                                                 patch_center)
                        segmentation_new_res_patch[segmentation_new_res_patch == 0] = prediction_patch[
                                                                                          segmentation_new_res_patch == 0] * label

                        segmentation = self.put_patch_in_full_mask(segmentation, predicted_segmentation_vertebrae_float,
                                                              patch_center, header, header_flipped['spacing'], new_shape, shape_flipped, label)

                        predicted_segmentation_discs = retain_largest_components(predicted_segmentation_discs,
                                                                                 labels=(True,), background=False, n=1)
                        segmentation_discs_new_res = self.put_patch_in_full_mask(segmentation_discs_new_res,
                                                                            predicted_segmentation_discs,
                                                                            patch_center, header, header_flipped['spacing'], new_shape, shape_flipped, label, resample=False)

                        predicted_segmentation_discs_float = retain_largest_components(predicted_segmentation_discs_float,
                                                                                       labels=(True,), background=False,
                                                                                       n=1)
                        segmentation_discs = self.put_patch_in_full_mask(segmentation_discs, predicted_segmentation_discs_float,
                                                                    patch_center, header, header_flipped['spacing'], new_shape, shape_flipped, label)

                        segmentation_sc = self.put_patch_in_full_mask(segmentation_sc, predicted_segmentation_sc_float,
                                                                 patch_center, header, header_flipped['spacing'], new_shape, shape_flipped, 1)

                        # Store classification of the detected vertebra
                        if predicted_completeness:
                            completely_visible_vertebrae.add(label)

                        # Store predicted label
                        labels[label] = np.clip(predicted_label, 1, self.max_vertebrae)

                        # Increase/reset counters
                        n_detected_vertebrae += 1
                        n_steps_vertebra = 0
                else:
                    # Did not converge, keep moving toward the center of the vertebra
                    n_steps_vertebra += 1

                    if n_steps_vertebra > self.max_steps:
                        # Move to center between two previous positions when the number of maximum steps is reached
                        patch_center = np.round(
                            (patch_center + new_patch_center) / 2
                        ).astype(int)
                    else:
                        patch_center = new_patch_center
            elif n_detected_vertebrae > 0:
                # No fragment of vertebral bone found, but we had found at least one vertebra before
                print(" > Lost trail")
                break
            else:
                # No fragment of vertebral bone found, keep moving in a sliding window fashion
                entire_image_searched = False

                if self.traversal_direction == "up":
                    for i in range(3):
                        patch_center[i] += self.patch_shape[i] // 2
                        if patch_center[i] <= scan_shape[i] - 1:
                            break
                        elif i == 2:
                            entire_image_searched = True
                        else:
                            patch_center[i] = starting_coordinates[i]
                else:
                    for i in range(3):
                        patch_center[i] -= self.patch_shape[i] // 2
                        if patch_center[i] >= 0:
                            break
                        elif i == 2:
                            entire_image_searched = True
                        else:
                            patch_center[i] = starting_coordinates[i]

                if entire_image_searched:
                    print(" > Reached end of scan")
                    break

        print(' > Traversal: {:.1f} s'.format(clock.elapsed()))

        # Determine most plausible labeling
        final_labels = []
        if n_detected_vertebrae > 0:
            for internal_label in labels:
                # Replace raw prediction with a probability vector
                probabilities = [0] * self.max_vertebrae

                # Incomplete vertebrae do not get to vote since their predictions might be incorrect
                if internal_label in completely_visible_vertebrae or len(completely_visible_vertebrae) == 0:
                    raw_prediction = labels[internal_label]
                    predicted_label = int(np.round(raw_prediction))

                    importance = 1

                    if predicted_label >= raw_prediction:
                        probabilities[predicted_label - 1] = (1 - (predicted_label - raw_prediction)) * importance
                        probabilities[predicted_label - 2] = predicted_label - raw_prediction
                    else:
                        probabilities[predicted_label - 1] = (1 - (raw_prediction - predicted_label)) * importance
                        probabilities[predicted_label] = raw_prediction - predicted_label

                labels[internal_label] = probabilities

            configurations = dict()

            internal_labels = list(sorted(np.unique(segmentation[segmentation > 0])))
            if self.traversal_direction == "up":
                internal_labels = internal_labels[::-1]

            for first_vertebra_label in range(
                1, self.max_vertebrae - n_detected_vertebrae + 2
            ):
                sum_probs = 0
                for label, internal_label in zip(
                    range(
                        first_vertebra_label,
                        first_vertebra_label + n_detected_vertebrae,
                    ),
                    internal_labels,
                ):
                    sum_probs += labels[internal_label][label - 1]
                configurations[first_vertebra_label] = sum_probs / n_detected_vertebrae

            best_configuration = max(configurations.items(), key=lambda item: item[1])
            first_label = best_configuration[0]
            print(
                "Best label configuration: {}-{} ({})".format(
                    first_label,
                    first_label + n_detected_vertebrae - 1,
                    best_configuration[1],
                )
            )

            labels = list(range(first_label, first_label + n_detected_vertebrae))
            lut = {old: new for old, new in zip(internal_labels, labels)}
            lut[0] = 0
            replace_labels = np.vectorize(lambda v: lut[v])
            segmentation = replace_labels(segmentation).astype('int16')
            segmentation_discs = replace_labels(segmentation_discs).astype('int16')

            print(completely_visible_vertebrae)
            print(lut)

            # Remove or relabel incompletely visible vertebrae
            completely_visible_vertebrae = set(
                lut[v] for v in completely_visible_vertebrae
            )
            for label in np.unique(segmentation[segmentation > 0]):
                if label in completely_visible_vertebrae:
                    final_labels.append(int(label))
                else:
                    new_label = 100 + label
                    segmentation[segmentation == label] = new_label
                    final_labels.append(int(new_label))

        print(' > LabelING: {:.1f} s'.format(clock.elapsed()))

        # Merge masks together
        total_segmentation = np.copy(segmentation)
        for label in labels:
            total_segmentation[np.logical_and(segmentation_discs == label, total_segmentation == 0)] = label + 200

        segmentation_sc = retain_largest_components(segmentation_sc, labels=(True,), background=False, n=1)
        total_segmentation[np.logical_and(segmentation_sc > 0, total_segmentation == 0)] = 100

        original_header['spacing'] = original_spacing
        print(header_flipped['direction'])
        print(original_header['direction'])

        if image_flipped:
            total_segmentation, _ = align_mask_with_image(total_segmentation, header_flipped, original_shape,
                                                          original_header)

        # Generate screenshot
        if screenshot_generator is not None:
            screenshot = screenshot_generator(original_image, total_segmentation)
            makedirs(path.dirname(screenshot_file), exist_ok=True)
            screenshot.save(screenshot_file)

        # Resample back to original resolution
        with TemporaryDirectory() as tmpdir:
            utils.remove_low_intensity_surface_voxels(image, total_segmentation, surface_erosion_threshold)
            tmpfile = path.join(tmpdir, 'segmentation.mha')
            write_image(tmpfile, total_segmentation, original_header)


        print(' > Resampling to original resolution: {:.1f} s'.format(clock.elapsed()))
        return final_labels, total_segmentation, original_header
