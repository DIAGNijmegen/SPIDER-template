FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.8-pt1.10-v2 AS base

RUN pip3 install --no-cache-dir rsgt

# Switch to non-root account
USER user


FROM base AS processor

# Copy data (scripts etc)
COPY --chown=user processor/README.md /app/README.md
COPY --chown=user processor/src/ /app/src/
COPY --chown=user processor/data /app/data/

# Define requirements
LABEL processor.cpus=1
LABEL processor.memory=24G
LABEL processor.gpu_count=1
LABEL processor.gpu.memory=8G

# Define entry point
WORKDIR /app/src
ENTRYPOINT ["python3", "SegmentSpine.py"]

# ----------------------------------------------------------------------------------------------------

FROM processor AS app

# Define sensible default arguments to match grand-challenge.org algorithm requirements
CMD ["--nometadata", "--imagedir", "/input/images/sagittal-spine-mri", "--maskdir", "/images/sagittal-spine-mr-segmentation", "--model", "up-mr-FinalSpineSegmentationDataset"]

# ----------------------------------------------------------------------------------------------------

