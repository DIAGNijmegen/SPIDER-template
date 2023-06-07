VERSION = $(shell python3 devel/config.py)

all: build push
devel: build_devel push_devel
interactive: build_interactive push_interactive
processor: build_processor push_processor
app: build_app push_app export_app
app_multitask: build_app_multitask push_app_multitask export_app_multitask

# Building docker images
build: build_devel build_processor build_app build_app_multitask build_interactive

build_devel:
	docker build \
		--target devel \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:devel \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:devel-$(VERSION) \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:latest \
		.

build_interactive:
	docker build \
		--target interactive \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:interactive \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:interactive-$(VERSION) \
		.

build_processor:
	docker build \
		--target processor \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:processor \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:processor-$(VERSION) \
		.

build_apps: build_app build_app_multitask

build_app:
	docker build \
		--target app \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:app \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:app-$(VERSION) \
		.

build_app_multitask:
	docker build \
		--target app-multitask \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:app-multitask \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:app-multitask-$(VERSION) \
		.

# Pushing docker images to the repository
push: push_devel push_processor push_app push_app_multitask push_interactive

push_devel:
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:devel
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:devel-$(VERSION)
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:latest

push_interactive:
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:interactive
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:interactive-$(VERSION)

push_processor:
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:processor
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:processor-$(VERSION)

push_apps: push_app push_app_multitask

push_app:
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:app
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:app-$(VERSION)

push_app_multitask:
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:app-multitask
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:app-multitask-$(VERSION)

# Export images
export_apps: export_app export_app_multitask

export_app:
	docker save doduo1.umcn.nl/nikolas/vertebra-segmentation:app-$(VERSION) | xz -c -T8 > "vertebra-segmentation-$(VERSION).tar.xz"

export_app_multitask:
	docker save doduo1.umcn.nl/nikolas/vertebra-segmentation:app-multitask-$(VERSION) | xz -c -T8 > "vertebra-segmentation-multitask-$(VERSION).tar.xz"

export_processor:
	docker tag doduo1.umcn.nl/nikolas/vertebra-segmentation:app-$(VERSION) radboudumc.nl/diag/vertebra-segmentation:processor
	docker save radboudumc.nl/diag/vertebra-segmentation:processor | xz -c -T8 > "vertebra-segmentation-processor-$(VERSION).tar.xz"

# Separate build instructions for MIDL 2020 processor to avoid rebuilding this all the time
MIDL2020:
	docker build \
		--target app-MIDL2020 \
		--tag doduo1.umcn.nl/nikolas/vertebra-segmentation:app-MIDL2020 \
		.
	docker push doduo1.umcn.nl/nikolas/vertebra-segmentation:app-MIDL2020
