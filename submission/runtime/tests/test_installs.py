import logging
import importlib

logging.getLogger("").setLevel(logging.INFO)

logging.info("Testing if Python packages can be loaded correctly.")

packages = [
    "fastai",
    "lightgbm",
    "pandas",
    "dotenv",
    "numpy",
    "torch",  # pytorch
    "sklearn",  # scikit-learn
    "scipy",
    "tensorflow",
    "xgboost",
    # ADD ADDITIONAL REQUIREMENTS BELOW HERE #
    ##########################################
    "tensorflow_addons",
    "keras",
    "cloudpickle",
    "tsfresh",
    "pytorch_lightning",
    "catboost"
]

for package in packages:
    logging.info("Testing if {} can be loaded...".format(package))
    importlib.import_module(package)

logging.info("All required packages successfully loaded.")
