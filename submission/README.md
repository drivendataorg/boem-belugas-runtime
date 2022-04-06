# BOEM Belugas Challenge

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue) [![Docker Image](https://img.shields.io/badge/Docker%20image-latest-green)](https://hub.docker.com/r/drivendata/noaa-competition/tags?page=1&name=latest)

Welcome to the runtime repository for the [BOEM Belugas Where's Whale-do](https://www.drivendata.org/competitions/96/competition-beluga-whales/) competition.

This repository contains the definition of the environment where your code submissions will run. It specifies both the operating system and the software packages that will be available to your solution.

This repository has two primary uses for competitors:

 - **Testing your code submission**: It lets you test your `submission.zip` file with a locally running version of the container so you don't have to wait for it to process on the competition site to find programming errors.
 - **Requesting new packages in the official runtime**: It lets you test adding additional packages to the official runtime [Python](https://github.com/drivendataorg/boem-belugas-runtime/blob/master/runtime/py.yml) environment. (The official runtime uses **Python 3.8.5**) You can then submit a PR to request compatible packages be included in the official container image.

Refer to the [code submission page](https://www.drivendata.org/competitions/96/competition-beluga-whales/page/482/) for details on what and how to submit and the data constraints of the environment.

 ----

### [Getting started](#0-getting-started)
 - [Prerequisites](#prerequisites)
 - [Quickstart](#quickstart)
### [Testing your submission locally](#1-testing-your-submission-locally)
 - [Implement your solution](#implement-your-solution)
 - [Example benchmark submission](#example-benchmark-submission)
 - [Running your submission](#running-your-submission)
 - [Reviewing the logs](#reviewing-the-logs)
### [Updating the runtime packages](#2-updating-the-runtime-packages)
 - [Adding new Python packages](#adding-new-python-packages)
 - [Testing new dependencies](#testing-new-dependencies)
 - [Opening a pull request](#opening-a-pull-request)

----

## (0) Getting started

### Prerequisites

Make sure you have the prerequisites installed.

 - A clone or fork of this repository
 - [Docker](https://docs.docker.com/get-docker/)
 - At least ~X GB of free space for both the training images and the Docker container images
 - GNU make (optional, but useful for using the commands in the Makefile)

### Quickstart

First, **download the data** from the competition [download page](https://www.drivendata.org/competitions/96/competition-beluga-whales/data/)
and put each file in the `data/` folder. After you get the data, you should have
these files:

```
data/
├── images
│     ├── IM00001.jpg
│     ├── IM00002.jpg
│     ├── IM00003.jpg
│       ...
├── metadata.csv
```

To test out the full execution pipeline, make sure docker is running and then run the following commands in order in the terminal. These will get the Docker
images, zip up an example submission script, and run the submission on your locally running version of the container.

```bash
make pull
make pack-benchmark
make test-submission
```

You should find a log from the run at `submission/log.txt`:

---

Running `make` at the terminal will tell you all the commands available in the repository:

```
$ make

Settings based on your machine:
SUBMISSION_IMAGE=72731633a8c5   # ID of the image that will be used when running test-submission

Available competition images:
drivendata/noaa-competition:local (72731633a8c5); drivendata/noaa-competition:latest (c6b3812166b2);

Available commands:

build               Builds the container locally
debug-container     Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
export-requirements Export the conda environment YAML from the container
pack-benchmark      Creates a submission/submission.zip file from whatever is in the "benchmark" folder
pull                Pulls the official container tagged latest from Docker hub
resolve-requirements Resolve the dependencies inside the container and write an environment YAML file on the host machine
test-container      Ensures that your locally built container can import all the Python packages successfully when it runs
test-submission     Runs container with submission/submission.zip as your submission and data as the data to work with
unpin-requirements  Remove specific version pins from Python conda environment YAML
```

To find out more about what these commands do, keep reading! :eyes:

## (1) Testing your submission locally

Your submission will run inside a Docker container, a virtual operating system that allows for a consistent software
environment across machines. This means that if your submission successfully runs in the container on your local machine,
you can be pretty sure it will successfully run when you make an official submission to the DrivenData site.

In Docker parlance, your computer is the "host" that runs the container. The container is isolated from your host machine,
with the exception of the following directories:

 - the `data` directory on the host machine is mounted in the container as a read-only directory `/codeexecution/data`
   **Note:**: you are not going to read this data directly, in fact, **attempting to do so is against the rules**.
   Your job is to write a function that can take seven days of data and make predictions with no additional context.
 - the `submission` directory on the host machine is mounted in the container as `/codeexecution/submission`

When you make a submission, the code execution platform will unzip your submission assets to the `/codeexecution` folder.
**This must result in a `predict.py` in the `/codeexecution`.**

### Implement your solution

In order to test your code submission, you will need a code submission! 

**Your only job is to replace the function `predict_dst` in [`benchmark/predict.py`](https://github.com/drivendataorg/noaa-runtime/blob/0baa35c6160e200bea5d4bc32029eabf49ae4957/benchmark/predict.py#L5)** with logic that will take seven days worth of
data (see the docstring) and output two predictions: one for the current hour (t0) and one for the next hour.
See the extensive description on the [code submission page](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/page/288/).

Keep in mind that your submission will not have access to the internet, so everything it needs to run must be provided
in the `submission.zip` you create. (You _are_ permitted to write intermediate files to `/codeexecution/submission`.)

**Note: You will implement all of your training and experiments on your machine. It is highly recommended that you use
the same package versions that are in the runtime [py.yml](runtime/py.yml). They can be installed with `conda`.**

### Example benchmark submission

We wrote a benchmark in Python to serve as a concrete example of a submission. Use `make pack-benchmark` to create the
benchmark submission from the source code. The command zips everything in the `benchmark` folder and saves the zip
archive to `submission/submission.zip`. To prevent losing your work, this command will not overwrite an existing submission.
To generate a new submission, you will first need to remove the existing `submission/submission.zip`.

### Running your submission

Now you can make sure your submission runs locally prior to submitting it to the platform. Make sure you have the
[prerequisites](#prerequisites) installed. Then, run the following command to download the official image:

```bash
make pull
```

Again, make sure you have packed up your solution in `submission/submission.zip` (or generated the sample submission
with `make pack-benchmark`), then try running it:

```bash
make test-submission
```

This will start the container, mount the local data and submission folders as folders within the container, and follow
the same steps that will run on the platform to unpack your submission and run your code. It should not take more than 5 minutes to run with the provided benchmark.

### Reviewing the logs

When you run `make test-submission` the logs will be printed to the terminal. They will also be written to the
`submission` folder as `log.txt`. You can always review that file and copy any versions of it that you want from the
`submission` folder. The errors there will help you to determine what changes you need to make so your code executes
successfully.

## (2) Updating the runtime packages

We accept contributions to add dependencies to the runtime environment. To do so, follow these steps:

1. Fork this repository
2. Make your changes
3. Test them and commit using git
4. Open a pull request to this repository

If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://guides.github.com/activities/forking/).

### Adding new Python packages

We use [conda](https://docs.conda.io/en/latest/) to manage Python dependencies. Add your new dependencies to `runtime/py.yml`.
Please also add your dependencies to `runtime/tests/test-installs.py`, below the line `## ADD ADDITIONAL REQUIREMENTS BELOW HERE ##`.

Your new dependency should follow the format in the yml and be pinned to a particular version of the package and build with conda.

### Testing new dependencies

Test your new dependency locally by recreating the relevant conda environment using the `py.yml` file. Try activating
that environment and loading your new dependency. Once that works, you'll want to make sure it works within the container
as well. To do so, you can run:

```
make test-container
```

Note: this will run `make build` to create the new container image with your changes automatically, but you could also do it manually.

This will build a local version of the container and then run the import tests to make sure the relevant libraries can
all be successfully loaded. This must pass before you submit a pull request to this repository to update the requirements.
If it does not, you'll want to figure out what else you need to make the dependencies happy.

If you have problems, the following command will run a bash shell in the container to let you interact with it. Make sure
to activate the `conda` environment (e.g., `source activate py`) when you start the container if you want to test
the dependencies!

```
make debug-container
```

### Opening a pull request

After making and testing your changes, commit your changes and push to your fork. Then, when viewing the repository on
github.com, you will see a banner that lets you open the pull request. For more detailed instructions, check out
[GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and
run the tests in `runtime/tests`. These tests take ~30 minutes to run through, and may take longer if your build is
queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.

You may be asked to submit revisions to your pull request if the tests fail, or if a DrivenData team member asks for
revisions. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.

---

## Good luck; have fun!

Thanks for reading! Enjoy the competition, and [hit up the forums](https://community.drivendata.org/) if you have any questions!
