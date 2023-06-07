Vertebra segmentation and identification with iterative FCNs
============================================================

**Processor**

PyTorch implementation of the inference part of the paper:

> N. Lessmann et al.,
> "Iterative fully convolutional neural networks for automatic vertebra segmentation and identification",
> Medical Image Analysis 53 (2019), pp. 142-155, https://doi.org/10.1016/j.media.2019.02.005

Inputs
------

The processor by default requires a metadata file, but can also be used with only a folder with image files.

The metadata file should be located in `/input/metadata.json` and needs to have the following format:

```
[
  {
    "entity": "1.2.840.113654.2.55.19...",
    "model": "up-ct-60",
    "surface_erosion_threshold": 200
  },
  {
    "entity": "1.2.840.113654.2.55.23...",
    "model": "up-mr-23"
  }
]
```

The entry `model` defines which set of weights is used to process the scan, and determines also the direction of
traversal (upwards or downwards). This parameter as well as the surface erosion threshold are optional.

Images need to be oriented according to the DICOM coordinate system (slices from feet to head) and stored
as mhd/mha in `/input/images`. The files need to be named identical to the "entity" name from the metadata
json file.

If no separate metadata file exists, the flag `-nometadata` can be used to process all mha/mhd files in the image
directory with default settings.

Outputs
-------

Vertebra segmentation masks are stored in `/output/vertebra_masks`. Vertebrae are labeled 1 (C1) to 24 (L5).

The format of the results json file is illustrated by the following example:

```
[
  {
    "entity": "1.2.840.1136...",
    "error_messages": [],
    "metrics": {
      "vertebra_mask": "filepath:/output/vertebra_masks/1.2.840.1136....mha"
    }
  }
]
```

The process returns 0 if at least one scan was (partially) successfully processed, otherwise 1.

Hardware requirements
---------------------

* 1 GPU with at least 8 GB memory
* Around 8-12 GB system memory
