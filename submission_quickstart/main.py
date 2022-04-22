from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


ROOT_DIRECTORY = Path("/code_execution")
PREDICTION_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"


class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, metadata):
        self.metadata = metadata
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, image_id):
        image = Image.open(DATA_DIRECTORY / self.metadata.loc[image_id].path).convert(
            "RGB"
        )
        image = self.transform(image)
        label = self.metadata.loc[image_id].whale_id
        sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


def main():
    # load competition data
    query_scenarios = pd.read_csv(
        DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id"
    )
    metadata = pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
    # instantiate data loader and pretrained model
    dataset = ImagesDataset(metadata)
    model = torch.load("model.pth")

    # precompute embeddings for all image ids across all queries
    all_img_ids = []
    for row in query_scenarios.itertuples():
        all_img_ids.extend(pd.read_csv(row.queries_path).query_image_id)
        all_img_ids.extend(pd.read_csv(row.database_path).database_image_id)

    embeddings = {}
    for img_id in all_img_ids:
        embeddings[img_id] = (
            model(dataset[img_id]["image"].unsqueeze(0)).squeeze(0).detach().numpy()
        )
    embeddings = pd.DataFrame.from_dict(embeddings, orient="index")

    # process all scenarios
    results = []
    for row in query_scenarios.itertuples():
        qry_df = pd.read_csv(row.queries_path)
        db_img_ids = pd.read_csv(row.database_path).database_image_id.values
        db_embeddings = embeddings.loc[db_img_ids]
        # predict matches for each query in this scenario
        for qry in qry_df.itertuples():
            sims = cosine_similarity(
                embeddings.loc[[qry.query_image_id]], db_embeddings
            )[0]
            sorter = np.argsort(-sims)
            top20 = db_img_ids[sorter][1:21]  # [1:21] drops nearest match which is the qry img itself
            scores = sims[sorter][1:21]
            qry_result = pd.DataFrame(
                {"query_id": qry.query_id, "database_image_id": top20, "score": scores}
            )
            results.append(qry_result)

    submission = pd.concat(results)
    submission.to_csv(PREDICTION_FILE, index=False)


if __name__ == "__main__":
    main()
