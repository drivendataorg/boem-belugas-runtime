.PHONY: build pull pack-benchmark pack-submission test-submission

# ================================================================================================
# Settings
# ================================================================================================

REPO = drivendata/belugas-competition
REGISTRY_IMAGE = boembelugas.azurecr.io/${REPO}:latest
LOCAL_IMAGE = ${REPO}:local

# if not TTY (for example GithubActions CI) no interactive tty commands for docker
ifneq (true, ${GITHUB_ACTIONS_NO_TTY})
TTY_ARGS = -it
endif

# To run a submission, use local version if that exists; otherwise, use official version
# setting SUBMISSION_IMAGE as an environment variable will override the image
SUBMISSION_IMAGE ?= $(shell docker images -q ${LOCAL_IMAGE})
ifeq (,${SUBMISSION_IMAGE})
SUBMISSION_IMAGE := $(shell docker images -q ${REGISTRY_IMAGE})
endif

# Give write access to the submission folder to everyone so Docker user can write when mounted
_submission_write_perms:
	chmod -R 0777 submission/

# ================================================================================================
# Commands for building the container if you are changing the requirements
# ================================================================================================

## Builds the container locally
build:
	echo ${LOCAL_IMAGE}
	docker build -t ${LOCAL_IMAGE} runtime

## Ensures that your locally built container can import all the Python packages successfully when it runs
test-container: build _submission_write_perms
	docker run \
		${TTY_ARGS} \
		--mount type=bind,source="$(shell pwd)"/runtime/run-tests.sh,target=/run-tests.sh,readonly \
		--mount type=bind,source="$(shell pwd)"/runtime/tests,target=/tests,readonly \
		${LOCAL_IMAGE} \
		/bin/bash -c "bash /run-tests.sh"

## Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
debug-container: build _submission_write_perms
	docker run \
		--mount type=bind,source="$(shell pwd)"/data,target=/codeexecution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/codeexecution/submission \
		--shm-size 8g \
		-it \
		${LOCAL_IMAGE} \
		/bin/bash

# ================================================================================================
# Commands for testing that your submission.zip will execute
# ================================================================================================

## Pulls the official container from Azure Container Registry
pull:
	docker pull ${REGISTRY_IMAGE}

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
		${TTY_ARGS} \
		--network none \
		--mount type=bind,source="$(shell pwd)"/data,target=/codeexecution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/codeexecution/submission \
	   	--shm-size 8g \
		${REGISTRY_IMAGE}

## Delete temporary Python cache and bytecode files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

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