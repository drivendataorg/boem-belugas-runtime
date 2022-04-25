### If you haven't already done so, start by reading the [Code Submission Format](https://www.drivendata.org/competitions/96/beluga-whales/page/482/) page on the competition website.


# Where's Whale-do: Code execution runtime

![Python 3.9.7](https://img.shields.io/badge/Python-3.9.7-blue) [![Docker Image](https://img.shields.io/badge/Docker%20image-latest-green)](https://hub.docker.com/r/drivendata/noaa-competition/tags?page=1&name=latest)

Welcome to the runtime repository for the [Where's Whale-do?](https://www.drivendata.org/competitions/96/beluga-whales/) beluga photo-identification challenge!

This repository contains the definition of the environment where your code submissions will run. It specifies both the operating system and the software packages that will be available to your solution.

This repository has three primary uses for competitors:

1. 💡 **Example solutions**: You can find two examples that will help you develop your own solution.
    - **[Quickstart example](TBD):** A minimal code example that runs succesfully in the runtime environment and outputs a properly formatted submission CSV. This will generate random predictions, so unfortunately you won't win the competition with this example, but you can use it as a guide for bringing in your own work and generating a real submission.
    - **Benchmark solution:** This example implements the benchmark solution based on the [benchmark blog post](TBD).
2. 🔧 **Test your submission**: Test your submission with a locally running version of the container to discover errors before submitting to the competition site.
3. 📦 **Request new packages in the official runtime**: Since the Docker container will not have network access, all packages must be pre-installed. If you want to use a package that is not in the runtime environment, make a pull request to this repository.

 ----

### [Quickstart](#quickstart)
 - [Prerequisites](#prerequisites)
 - [Download the data](#download-the-data)
 - [Run Make commands](#run-make-commands)
### [Developing your own submission](#developing-your-own-submission)
 - [Steps](#steps)
 - [Logging](#logging)
 - [Scoring your submission](#scoring-your-submission)
### [Additional information](#additional-information)
 - [Runtime network access](#runtime-network-access)
 - [CPU and GPU](#cpu-and-gpu)
 - [Make commands](#make-commands)
 - [Updating runtime packages](#updating-runtime-packages)


----

## Quickstart

This section guides you through the steps to generate a simple but valid submission for the competition.

### Prerequisites

First, make sure you have the prerequisites installed.

 - A clone or fork of this repository
 - At least 13 GB (_todo: double check with final image_) of free space for both the training images and the Docker container images.
 - [Docker](https://docs.docker.com/get-docker/)
 - [GNU make](https://www.gnu.org/software/make/) (optional, but useful for running commands in the Makefile)

### Download the data

First, download the data from the competition [download page](https://www.drivendata.org/competitions/96/beluga-whales/data/)
and copy each file into the project `data` folder. Once everything is downloaded and in the right location, it should look something like this:

```
data/                         # Competition data directory
├── images/                   # Directory containing all the query and database images
│      ├── train0000.jpg
│      ├── train0001.jpg
│      ├── train0002.jpg
│      └── ...
├── metadata.csv              # CSV file with image metadata (image dimensions, viewpoint, date)
├── databases/                # Directory containing the database image IDs for each scenario
│      ├── scenario01-example.txt
│      ├── scenario02-example.txt
│      └── scenario03-example.txt
└── queries/                  # Directory containing the query IDs and image IDs for each scenario
       ├── scenario01-example.csv
       ├── scenario02-example.csv
       └── scenario03-example.csv
```

Later in this guide, when we launch a Docker container from your computer (or the "host" machine), the `data` directory on your host machine will be mounted as a read-only directory in the container as `code_execution/data`. In the runtime, your code will then be able to access all the competition data at `code_execution/data`.

#### What's in `data`

Let's briefly review the contents of `data` and what they're used for:

* **`images`** directory contains all of the image files (.jpg).
* **`metadata.csv`** is a CSV file containing metadata (image dimensions, viewpoint, etc) for each image.
* **`databases`** directory contains a text file for each _scenario_. Each text file contains the image IDs for all images in its scenario _database_. For example, `scenario01-example.txt` might look something like this, where `train0123` is an image ID:

    ```
    train0123
    train0321
    train0333
    ...
    ```

* **`queries`** directory contains a CSV file for each _scenario_. Each CSV file contains the query IDs and query _image_ IDs for each query in its scenario:

    ```
    query_id               query_image_id
    scenario01-train1111   train1111
    scenario01-train1112   train1112
    scenario01-train1122   train1122
    ...                    ...
    ```

The core task you will be evaluated on in this competition requires you to **generate image rankings for each query in each scenario.** This will be done during code execution using the `main.py` script, which iterates through each scenario and runs inference to generate your image rankings.

Here is simplified pseudo-code showing what `main.py` does for illustrative purposes:

```python
submission = []

for scenario in scenarios:
   queries = load(f'/data/queries/{scenario}.csv')
   database = load(f'/data/databases/{scenario}.txt')

   for query in queries:
      image_rankings = run_inference(query, database) # <--- your code here
      submission.append(image_rankings)
```

An example `main.py` script is provided in this repository for you to get started. You are free to modify `main.py` as needed provided that you are otherwise adhering to the [competition rules](https://www.drivendata.org/competitions/96/beluga-whales/rules/). Your main focus in this competition will be on developing the `run_inference` command above (or a similar function you've defined) such that it can produce good matches between the query and database images of beluga whales.

But we'll get around to all that in due time. For now, let's continue with the quickstart and generate a trivial submission of random predictions so that we can see the entire pipeline in action.

### Run Make commands

To test out the full execution pipeline, make sure Docker is running and then run the following commands in the terminal:

1. **`make pull`** pulls the latest official Docker image from the container registry ([Azure](https://azure.microsoft.com/en-us/services/container-registry/)). You'll need an internet connection for this.
2. **`make pack-quickstart`** zips the contents of the `submission_quickstart` directory and saves it as `submission/submission.zip`. This is the file containing your code that you will upload to the DrivenData competition site for code execution. But first we'll test that everything looks good locally (see next step).
3. **`make test-submission`** will do a test run of your submission, simulating what happens during actual code execution. This command runs the Docker container with the requisite host directories mounted, and executes `main.py` to produce a CSV file with your image rankings at `submission/submission.csv`.

```bash
make pull
make pack-quickstart
make test-submission
```

🎉 **Congratulations!** You've just completed your first test run for the BOEM Belugas competition. If this were a real submission, you would upload the `submission.zip` file from step 2 above to the competition [Submissions page](https://www.drivendata.org/competitions/96/beluga-whales/submissions/).

Finally, there's one last step that takes place at the end of code execution and which we haven't described yet: the `submission.csv` that is written out during code execution will get **scored** automatically using the [competition scoring metric](https://www.drivendata.org/competitions/96/beluga-whales/page/479/#performance_metric) to determine your rank on the leaderboard. We'll talk in more detail below about how you can also score your submissions locally (with the _publicly_ available labels) without using up your submissions budget.

----

## Developing your own submission

Now that you've gone through the quickstart example, let's talk about how to develop your own solution for the competition.

### Steps

This section provides instructions on how to develop and run your code submission locally using the Docker container. To make things simpler, key processes are already defined in the `Makefile`. Commands from the `Makefile` are then run with `make {command_name}`. The basic steps are:

```
make pull
make pack-submission
make test-submission
```

Let's walk through what you'll need to do, step-by-step. The overall process here is very similar to what we've already covered in the [Quickstart](#quickstart), but we'll go into more depth this time around.

1. **[Set up the prerequisites](#prerequisites)**

2. **[Download the data](#download-the-data)**

3. **Download the official competition Docker image:**

    ```bash
    $ make pull
    ```

4. ⚙️ **Save all of your submission files, including the required `main.py` script, in the `submission_src` folder of the runtime repository.** This is where the real work happens.
   * You are free to modify the `main.py` template we've provided, and you'll obviously want to add any code necessary to process the queries, cache results, and run inference. Just make sure that you adhere to the competition rules and you still produce a `submission.csv` in the correct format.
   * Also keep in mind that the runtime already contains a number of packages that might be useful for you ([cpu](https://github.com/drivendataorg/boem-belugas-runtime/tree/master/runtime/environment-cpu.yml) and [gpu](https://github.com/drivendataorg/boem-belugas-runtime/tree/master/runtime/environment-gpu.yml) versions). If there are other packages you'd like added, see the section below on [updating runtime packages](#updating-runtime-packages).
   * Finally, make sure any model weights or other files you need are also saved in `submission_src`.

5. **Create a `submission/submission.zip` file containing your code and model assets:**

    ```bash
    $ make pack-submission
    cd submission_src; zip -r ../submission/submission.zip ./*
      adding: main.py (deflated 50%)
    ```

6. **Launch an instance of the competition Docker image, running the same inference process that will take place in the official code execution runtime.** This will mount the requisite host directories on the Docker container, unzip `submission/submission.zip` into the root directory of the container, and then execute `main.py` to produce a CSV file with your image rankings at `submission/submission.csv`.

   ```
   $ make test-submission
   ```



> ⚠️ **Remember** that for local testing purposes, the `code_execution/data` directory is just a mounted version of what you have saved locally in this project's `data` directory. So you will just be using the publicly available training files for local testing. In the official code execution environment, `code_execution/data` will contain the _actual test data_, which no participants have access to, and this is what will be used to compute your score for the leaderboard.



### Logging

When you run `make test-submission` the logs will be printed to the terminal and written out to `submission/log.txt`. If you run into errors, use the `log.txt` to determine what changes you need to make for your code to execute successfully.


### Scoring your submission

We will provide a [metric script](TBD) to calculate the competition metric in the same way scores will be calculated in the DrivenData platform. To score your submission:

1. After running the above, verify that image rankings generated by your code are saved at `submission/submission.csv`.

2. Make sure that the metadata CSV file is saved in `data/metadata.csv`. This file contains the labels that we'll use for scoring.

3. Run `scripts/metric.py` on your predictions:
    ```bash
    # show usage instructions
    $ python scripts/metric.py --help
    ```

---
## Additional information

### Runtime network access

In the real competition runtime, all internet access is blocked. The local test runtime does not impose the same network restrictions. It's up to you to make sure that your code doesn't make requests to any web resources.

You can test your submission _without_ internet access by running `BLOCK_INTERNET=true make test-submission`.

### Downloading pre-trained weights

It is common for models to download pre-trained weights from the internet. Since submissions do not have open access to the internet, you will need to include all weights along with your `submission.zip` and make sure that your code loads them from disk and rather than the internet.


### CPU and GPU

The `make` commands will try to select the CPU or GPU image automatically by setting the `CPU_OR_GPU` variable based on whether `make` detects `nvidia-smi`.

You can explicitly set the `CPU_OR_GPU` variable by prefixing the command with:
```bash
CPU_OR_GPU=cpu <command>
```

**If you have `nvidia-smi` and a CUDA version other than 11**, you will need to explicitly set `make test-submission` to run on CPU rather than GPU. `make` will automatically select the GPU image because you have access to GPU, but it will fail because `make test-submission` requires CUDA version 11.
```bash
CPU_OR_GPU=cpu make pull
CPU_OR_GPU=cpu make test-submission
```

If you want to try using the GPU image on your machine but you don't have a GPU device that can be recognized, you can use `SKIP_GPU=true`. This will invoke `docker` without the `--gpus all` argument.

### Updating runtime packages

If you want to use a package that is not in the environment, you are welcome to make a pull request to this repository. If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://docs.github.com/en/get-started/quickstart/contributing-to-projects). The runtime manages dependencies using [conda](https://docs.conda.io/en/latest/) environments. [Here is a good general guide](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) to conda environments. The official runtime uses **Python 3.9.7** environments.

To submit a pull request for a new package:

1. Fork this repository.

2. Edit the [conda](https://docs.conda.io/en/latest/) environment YAML files, `runtime/environment-cpu.yml` and `runtime/environment-gpu.yml`. There are two ways to add a requirement:
    - Add an entry to the `dependencies` section. This installs from a conda channel using `conda install`. Conda performs robust dependency resolution with other packages in the `dependencies` section, so we can avoid package version conflicts.
    - Add an entry to the `pip` section. This installs from PyPI using `pip`, and is an option for packages that are not available in a conda channel.

    For both methods be sure to include a version, e.g., `numpy==1.20.3`. This ensures that all environments will be the same.

3. Locally test that the Docker image builds successfully for CPU and GPU images:

    ```sh
    CPU_OR_GPU=cpu make build
    CPU_OR_GPU=gpu make build
    ```

4. Commit the changes to your forked repository.

5. Open a pull request from your branch to the `main` branch of this repository. Navigate to the [Pull requests](https://github.com/drivendataorg/cloud-cover-runtime/pulls) tab in this repository, and click the "New pull request" button. For more detailed instructions, check out [GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

6. Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and running the tests in `runtime/tests`. These tests can take up to 30 minutes, and may take longer if your build is queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.

7. You may be asked to submit revisions to your pull request if the tests fail or if a DrivenData team member has feedback. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.


### Make commands

Running `make` at the terminal will tell you all the commands available in the repository:

```
Settings based on your machine:
SUBMISSION_IMAGE=f6961d910a89   # ID of the image that will be used when running test-submission

Available competition images:
drivendata/belugas-competition:cpu-local (f6961d910a89); drivendata/belugas-competition:gpu-local (916b2fbc2308);

Available commands:

build               Builds the container locally
clean               Delete temporary Python cache and bytecode files
interact-container  Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
pack-quickstart     Creates a submission/submission.zip file from the source code in submission_quickstart
pack-submission     Creates a submission/submission.zip file from the source code in submission_src
pull                Pulls the official container from Azure Container Registry
test-container      Ensures that your locally built container can import all the Python packages successfully when it runs
test-submission     Runs container using code from `submission/submission.zip` and data from `data/`
```

---

## Good luck! And have fun!

Thanks for reading! Enjoy the competition, and [hit up the forums](https://community.drivendata.org/c/belugas/86) if you have any questions!
