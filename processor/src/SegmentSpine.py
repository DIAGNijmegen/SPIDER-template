import sys
import json
from os import path, makedirs
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

from tiger.io import read_image, write_image
import utils


# Define parameters
parser = ArgumentParser(description="Vertebra segmentation pipeline")
parser.add_argument(
    "--imagedir",
    help="Filename of the image",
    default="/input/images/sagittal-spine-mri"
)
parser.add_argument(
    "--metadatafile",
    help="Filename of the metadata file",
    default="/input/metadata.json"
)
parser.add_argument(
    "--model",
    help="Model that is used to process the data",
    default="up-ct-jun2022"
)
parser.add_argument(
    "--outdir",
    help="Directory path for generated output (vertebra mask, etc.)",
    default="/output/images/sagittal-spine-mr-segmentation"
)
parser.add_argument(
    "--maskdir",
    help="Subdirectory of outdir in which segmentation masks are stored",
    default="/images/sagittal-spine-mr-segmentation"
)
parser.add_argument(
    "--extradir",
    help="Subdirectory of outdir in which additional segmentation masks are stored",
    default="sagittal-spine-mr-segmentation"
)
parser.add_argument(
    "--nometadata",
    help="Process all images in the input directory",
    action="store_true"
)
parser.add_argument(
    "--skipexisting",
    help="Skip images with existing segmentation mask",
    action="store_true"
)
parser.add_argument(
    "--cpu",
    help="Run on the CPU instead of the GPU",
    action="store_true"
)

args = parser.parse_args()

# Read metadata file
if args.nometadata:
    print("Getting image list from {}".format(args.imagedir))
    metadata = [{"entity": uid} for _, uid in utils.find_image_files(args.imagedir)]
else:
    print("Reading metadata from {}".format(args.metadatafile))

    if not path.exists(args.metadatafile):
        print("Metadata file does not exist", file=sys.stderr)
        exit(1)

    with open(args.metadatafile) as jsonfile:
        metadata = json.load(jsonfile)

print(" > {} images".format(len(metadata)))

# Construct pipeline
print("Preparing vertebra segmentation pipeline")
kwargs = {
    "device": "cpu" if args.cpu else "cuda"
}
# pipeline = SpineSegmentationPipeline(args.model, **kwargs)

if args.cpu:
    print("Running on CPU!")

# Iterate over all images and process them one by one
results = []
any_success = False

for scan in metadata:
    # Try to find the UID of the entity (this is crucial to be able to find related files)
    try:
        uid = scan["entity"]
    except (KeyError, TypeError):
        print('Invalid metadata, missing "entity" value, skipping entry', file=sys.stderr)
        continue

    error_messages = []
    result_message = "Algorithm failed"

    # Print out ID of the scan
    print("-" * 50)
    print("Processing {}".format(uid))

    # Check if result already exists
    overlay_file = path.join(args.outdir.strip(), args.maskdir.strip(), uid + ".mha")
    extra_overlay_file = path.join(args.outdir.strip(), args.extradir.strip(), uid + ".mha")

    if args.skipexisting and path.exists(overlay_file):
        print("Segmentation mask exists already, skipping...")
        continue

    try:
        # Load image
        image_file = utils.find_image_file(args.imagedir.strip(), uid)
        if image_file is None:
            raise ValueError(
                'Found no image file for basename "{}" in {}'.format(uid, args.imagedir)
            )

        # Run vertebra segmentation pipeline on the image
        print("Detecting vertebrae...")
        clock = utils.Timer()

        # implement your segmentation method here!
        original_image, original_header = read_image(image_file)
        # the variables that are saved, and therefor should be named accordingly are total_segmentation, and original_header

        print("Total runtime: {:.1f} s".format(clock.elapsed()))

        any_success = True
    except (RuntimeError, ValueError, MemoryError, IOError, FileNotFoundError) as e:
        try:
            error_message = e.message
        except AttributeError:
            error_message = str(e)

        print("Error: {}".format(error_message), file=sys.stderr)
        error_messages.append(error_message)
    finally:
        results.append({
            "entity": uid,
            "error_messages": error_messages,
            "result_message": result_message
        })
        # Overwrite results file
        outdir = args.outdir.strip()
        if not path.exists(outdir):
            makedirs(outdir)
            print(outdir)
        with open(path.join(outdir, "results.json"), "w") as jsonfile:
            json.dump(results, jsonfile, indent=2, separators=(",", ": "))

        output_dir = path.join('output', outdir, args.maskdir, 'segmentation.mha')
        print(output_dir)
        try:
            write_image(output_dir, total_segmentation, original_header)
        except PermissionError:
            print('got error, used the hardcoded outdir')
            output_dir = Path('/output/images/sagittal-spine-mr-segmentation/segmentation.mha')
            write_image(output_dir, total_segmentation, original_header)

print("Processed {} images".format(len(results)))
exit(0 if any_success else 1)
