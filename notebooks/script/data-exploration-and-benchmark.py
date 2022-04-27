#!/usr/bin/env python
# coding: utf-8

# ## Welcome




# ## Download the data
# 
# First, download the `images` and `metadata.csv` files from the [competition website](https://www.drivendata.org/competitions/96/beluga-whales/data/).
# 
# Save the files in the `data` directory so that your tree looks like this. 
# ```
# boem-belugas-runtime/             # This repository's root
# └── data/                         # Competition data directory
#     ├── databases/                # Directory containing the database image IDs for 
#     │      │                          each scenario
#     │      ├── scenario01.csv
#     │      └── scenario02.csv
#     ├── images/                   # Directory containing all the images
#     │      ├── train0001.jpg
#     │      ├── train0002.jpg
#     │      ├── train0003.jpg
#     │      └── ...
#     ├── queries/                  # Directory containing the query image IDs for 
#     │      │                          each scenario
#     │      ├── scenario01.csv
#     │      └── scenario02.csv
#     ├── metadata.csv              # CSV file with image metadata (image dimensions, 
#     │                                 viewpoint, date)
#     └── query_scenarios.csv       # CSV file that lists all test scenarios with paths
# ```
# 
# If you're working off a clone of this [runtime repository](https://github.com/drivendataorg/boem-belugas-runtime), you should already have copies of the `databases`, `queries` and `query_scenarios.csv` files.

# ## Explore the data
# First, let's load the metadata file.

from pathlib import Path

import pandas as pd

PROJ_DIRECTORY = Path.cwd().parent
DATA_DIRECTORY = PROJ_DIRECTORY / "data"

metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")


# ### Look at some sample images
# Let's begin by looking at some images with our regular old human eyes, before handing things over to the computer.
# 
# The function below shows a random sample of images (change `random_state` to get a new set) for a given viewpoint. 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images(viewpoint="top", random_state=1, metadata=metadata):
    # set plot layout depending on image viewpoint
    nrows, ncols = (1, 5) if viewpoint == "top" else (4, 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,8))
    # get a random sample of images
    sample = metadata[metadata.viewpoint == viewpoint].sample(nrows*ncols, random_state=random_state)
    # plot in grid
    for img_path, ax in zip(sample.path, axes.flat):
        img = mpimg.imread(DATA_DIRECTORY / img_path)
        ax.imshow(img)    


# Let's look at a random sample of "top" images, taken from overhead by drone. Note the differences in color, marks and scarring, as well as which regions of the body are visible. Also note that other factors like the water and lighting conditions can affect the quality of the image.

display_images("top", random_state=1)


display_images("top", random_state=2)


display_images("top", random_state=3)


# Most images in the dataset (92%) are taken from overhead like the ones above. But there are also some "lateral" images that are taken from vessels. These images have a viewpoint of either `"left"` or `"right"`.

display_images("left", random_state=1)


metadata.viewpoint.value_counts()


# ### Dataset summary
# In total, there are **5902 images** in the training dataset, and **788 unique whale IDs**. 
# 
# Whale IDs identify individual belugas, though an individual can have more than one whale ID over its lifetime due to changes in its appearance over time. Check out the [Problem Description](https://www.drivendata.org/competitions/96/beluga-whales/page/479/#whale_id) for a more detailed discussion of whale IDs, if you haven't already.

print(f"n images:    {len(metadata)}")
print(f"n whale IDs: {metadata.whale_id.nunique()}")


# #### Image dimensions
# The height and width of the images vary. "Top" images tend to be "tall", while lateral images tend to be "wide". Let's take a quick look at the distribution of image dimensions.

def display_image_dimensions(metadata, title=None):
    lim = max(metadata.width.max(), metadata.height.max())*1.1
    plt.figure(figsize=(5,5))
    plt.scatter(metadata.width, metadata.height, alpha=0.1)
    plt.ylim(0,lim)
    plt.ylabel('image height (pixels)')
    plt.xlim(0,lim)
    plt.xlabel('image width (pixels)')
    plt.title(title)


viewpoint_metadata = metadata[metadata.viewpoint == "top"]
display_image_dimensions(viewpoint_metadata, title="top images")


# Note that there are a handful of "extra wide" lateral images.

viewpoint_metadata = metadata[metadata.viewpoint != "top"]
display_image_dimensions(viewpoint_metadata, title="lateral images")


# #### Matches
# As noted above, we can use the `whale_id` data to identify images of the same whale.
# 
# Typically we'll have between 2-6 distinct images of most whales. But in some cases we may have just one image of a given whale, and in other cases we have many more (>100).

whale_id_counts = metadata.groupby("whale_id").size()

whale_id_counts.hist(bins=range(107), grid=False, figsize=(10,5), edgecolor="white")
plt.axvline(whale_id_counts.mean(), color='k', alpha=0.3)
plt.text(whale_id_counts.mean(), 100, s="mean: {:.1f}".format(whale_id_counts.mean()))
plt.xlabel("n images for whale ID")
plt.ylabel("count")

print(whale_id_counts.value_counts().sort_index().head(10))


# #### Match examples
# Here is a sample of images from `whale093`, which has the most images (106) of any individual in the dataset. Note the similarity in color and the distinctive notch in this whale's dorsal fin.

metadata_individual = metadata[metadata.whale_id == "whale093"]
display_images("top", metadata=metadata_individual, random_state=1)


# Let's look at some other sets of matching images for a given whale to get a sense for the visual features that make an individual beluga identifiable.
# 
# We'll restrict ourselves to "top" images for now to keep things simple, and we'll look at whales that have at least 5 images so we can get an idea for how these matches can vary.

whale_id_counts_top = metadata[metadata.viewpoint=="top"].groupby("whale_id").size()
whales_with_5_top_images = whale_id_counts_top[whale_id_counts_top >= 5]
whales_with_5_top_images.index


# The marks around the blowhole of this beluga are probably its most identifiable feature.

whale_id = whales_with_5_top_images.sample(1, random_state=1).index[0]
display_images("top", metadata=metadata[metadata.whale_id == whale_id])
print(whale_id)


# This beluga has some light scratches near the top of its dorsal fin, but they are very subtle.

whale_id = whales_with_5_top_images.sample(1, random_state=2).index[0]
display_images("top", metadata=metadata[metadata.whale_id == whale_id])
print(whale_id)


# The scratches along the back and some distinctive features around the blowhole distinguish this beluga.

whale_id = whales_with_5_top_images.sample(1, random_state=3).index[0]
display_images("top", metadata=metadata[metadata.whale_id == whale_id])
print(whale_id)


# #### Cropped images
# You may have noticed above that some images contain triangular regions of white space. This occurs because the original photos of these belugas are taken with a much wider field of view that may contain multiple belugas, before being passed through an auto-detection algorithm that identifies each individual beluga whale and draws a bounding box around it. 
# 
# When these bounding boxes happen to overlap with the edges of the original photo, the region of the box for which we have no data (it is beyond the original image's border) are encoded as white pixels.












