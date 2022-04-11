.PHONY: build pull pack-benchmark pack-submission test-submission

# ================================================================================================
# Settings
# ================================================================================================

REPO = drivendata/boem-belugas-competition
TAG = latest
IMAGE = ${REPO}:${TAG}

# if not TTY (for example GithubActions CI) no interactive tty commands for docker
ifneq (true, ${GITHUB_ACTIONS_NO_TTY})
TTY_ARGS = -it
endif

# To run a submission, use local version if that exists; otherwise, use official version
# setting SUBMISSION_IMAGE as an environment variable will override the image
SUBMISSION_IMAGE ?= $(shell docker images -q ${LOCAL_IMAGE})
ifeq (,${SUBMISSION_IMAGE})
SUBMISSION_IMAGE := $(shell docker images -q ${IMAGE})
endif

# Give write access to the submission folder to everyone so Docker user can write when mounted
_submission_write_perms:
	chmod -R 0777 submission/

# ================================================================================================
# Commands for building the container if you are changing the requirements
# ================================================================================================

## Builds the container locally
build:
	docker build -t ${IMAGE} runtime

# ================================================================================================
# Commands for testing that your submission.zip will execute
# ================================================================================================

## Pulls the official container from Azure Container Registry
pull:
	docker pull boembelugas.azurecr.io/${IMAGE}

## Creates a submission/submission.zip file from the benchmark source code in benchmark_src
pack-benchmark:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	cd benchmark_src; zip -r ../submission/submission.zip ./*

## Creates a submission/submission.zip file from the source code in submission_src
pack-submission:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	cd submission_src; zip -r ../submission/submission.zip ./*


## Runs container using code from `submission/submission.zip` and data from `data/`
test-submission: _submission_write_perms
# if submission file does not exist
ifeq (,$(wildcard ./submission/submission.zip))
	$(error To test your submission, you must first put a "submission.zip" file in the "submission" folder. \
	  If you want to use the benchmark, you can run `make pack-benchmark` first)
endif

# if container does not exist, error and tell user to pull or build
ifeq (${SUBMISSION_IMAGE},)
	$(error To test your submission, you must first run `make pull` (to get official container) or `make build` \
		(to build a local version if you have changes).)
endif
	docker run \
		--mount type=bind,source="$(shell pwd)"/data,target=/codeexecution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/codeexecution/submission \
		${IMAGE}


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help
.PHONY: help

define PRINT_HELP_PYSCRIPT
import re, sys

pattern = re.compile(r'^## (.*)\n(.+):', re.MULTILINE)
text = "".join(line for line in sys.stdin)
for match in pattern.finditer(text):
    help, target = match.groups()
    print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)