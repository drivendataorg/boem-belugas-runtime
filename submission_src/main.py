from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union, cast

from PIL import Image
from conduit.data.structures import MeanStd
from loguru import logger
import pandas as pd  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T  # type: ignore
from tqdm import tqdm  # type: ignore
from typing_extensions import Final, TypeAlias

from whaledo.models.artifact import load_model_from_artifact
from whaledo.models.base import Model
from whaledo.transforms import ResizeAndPadToSize

ROOT_DIRECTORY: Final[Path] = Path("/code_execution")
PREDICTION_FILE: Final[Path] = ROOT_DIRECTORY / "submission" / "submission.csv"
DATA_DIRECTORY: Final[Path] = ROOT_DIRECTORY / "data"
MODEL_PATH: Final[str] = "model.pt"
DEFAULT_IMAGE_SIZE: Final[int] = 256


class MeanStd(NamedTuple):
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


IMAGENET_STATS: Final[MeanStd] = MeanStd(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

ImageTform: TypeAlias = Callable[[Image.Image], Any]


class TestTimeWhaledoDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id and image tensors.
    """

    def __init__(
        self, metadata: pd.DataFrame, image_size: Optional[int] = DEFAULT_IMAGE_SIZE
    ) -> None:
        if image_size is None:
            image_size = DEFAULT_IMAGE_SIZE
        self.metadata = metadata
        transform_ls: List[ImageTform] = [
            ResizeAndPadToSize(DEFAULT_IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(*IMAGENET_STATS),
        ]
        self.transform = T.Compose(transform_ls)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Image.Image]]:
        image = Image.open(DATA_DIRECTORY / self.metadata.path.iloc[idx]).convert("RGB")
        image = self.transform(image)
        return {"image_id": self.metadata.index[idx], "image": image}

    def __len__(self) -> int:
        return len(self.metadata)


def main() -> None:
    logger.info("Starting main script")
    # load test set data and pretrained model
    query_scenarios = cast(
        pd.DataFrame, pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv", index_col="scenario_id")
    )
    metadata = cast(
        pd.DataFrame, pd.read_csv(DATA_DIRECTORY / "metadata.csv", index_col="image_id")
    )
    logger.info("Loading pre-trained model...")
    backbone, feature_dim, image_size = load_model_from_artifact(MODEL_PATH)
    model = Model(backbone=backbone, feature_dim=feature_dim)

    # we'll only precompute embeddings for the images in the scenario files (rather than all images), so that the
    # benchmark example can run quickly when doing local testing. this subsetting step is not necessary for an actual
    # code submission since all the images in the test environment metadata also belong to a query or database.
    scenario_imgs = []
    for row in query_scenarios.itertuples():
        scenario_imgs.extend(pd.read_csv(DATA_DIRECTORY / row.queries_path).query_image_id.values)
        scenario_imgs.extend(
            pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
        )
    scenario_imgs = sorted(set(scenario_imgs))
    metadata = metadata.loc[scenario_imgs]

    # instantiate dataset/loader and generate embeddings for all images
    dataset = TestTimeWhaledoDataset(metadata, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=16)
    embeddings = []
    model.eval()

    logger.info("Precomputing embeddings")
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch_embeddings = model(batch["image"])
        batch_embeddings_df = pd.DataFrame(
            batch_embeddings.detach().numpy(), index=batch["image_id"]
        )
        embeddings.append(batch_embeddings_df)

    embeddings = pd.concat(embeddings)
    logger.info(f"Precomputed embeddings for {len(embeddings)} images")
    logger.info("Generating image rankings")
    # process all scenarios
    results = []
    for row in query_scenarios.itertuples():
        # load query df and database images; subset embeddings to this scenario's database
        qry_df = cast(pd.DataFrame, pd.read_csv(DATA_DIRECTORY / row.queries_path))
        db_img_ids = cast(
            pd.DataFrame, pd.read_csv(DATA_DIRECTORY / row.database_path).database_image_id.values
        )
        db_embeddings = embeddings.loc[db_img_ids]

        # predict matches for each query in this scenario
        for qry in qry_df.itertuples():
            # get embeddings; drop query from database, if it exists
            qry_embedding = embeddings.loc[[qry.query_image_id]]
            _db_embeddings = db_embeddings.drop(qry.query_image_id, errors="ignore")

            prediction = model.predict(queries=qry_embedding, db=_db_embeddings, k=20)
            scores = pd.Series(prediction.scores)
            # append result
            qry_result = pd.DataFrame(
                {
                    "query_id": qry.query_id,
                    "database_image_id": scores.index,
                    "score": scores.values,
                }
            )
            results.append(qry_result)

    logger.info(f"Writing predictions file to {PREDICTION_FILE}")
    submission = pd.concat(results)
    submission.to_csv(PREDICTION_FILE, index=False)


if __name__ == "__main__":
    main()
