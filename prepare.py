from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]


class RentalsDataset(Dataset):
    """Return one feature vector and one scalar target per row."""

    def __init__(self, features, targets):
        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.targets = torch.tensor(np.asarray(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


@dataclass
class PreparedData:
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    preprocessor: ColumnTransformer
    train_dataset: RentalsDataset
    val_dataset: RentalsDataset
    test_dataset: RentalsDataset

    @property
    def n_features(self):
        return self.train_dataset.features.shape[1]


def prepare_data(seed=42):
    frame = make_rentals()

    # Split before fitting any preparation.
    train_frame, remaining = train_test_split(frame, test_size=0.4, random_state=seed)
    val_frame, test_frame = train_test_split(
        remaining, test_size=0.5, random_state=seed
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )

    train_features = preprocessor.fit_transform(train_frame.drop(columns=TARGET))
    val_features = preprocessor.transform(val_frame.drop(columns=TARGET))
    test_features = preprocessor.transform(test_frame.drop(columns=TARGET))

    return PreparedData(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        preprocessor=preprocessor,
        train_dataset=RentalsDataset(train_features, train_frame[TARGET]),
        val_dataset=RentalsDataset(val_features, val_frame[TARGET]),
        test_dataset=RentalsDataset(test_features, test_frame[TARGET]),
    )


def make_loaders(data, batch_size=64, seed=42):
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        data.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(data.val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data.test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data = prepare_data()
    train_loader, _, _ = make_loaders(data)
    features, targets = next(iter(train_loader))

    print("Training rows:", len(data.train_dataset))
    print("Validation rows:", len(data.val_dataset))
    print("Test rows:", len(data.test_dataset))
    print("Number of features:", data.n_features)
    print("Batch feature shape:", features.shape)
    print("Batch target shape:", targets.shape)
    print("Feature dtype:", features.dtype)
    print("Target dtype:", targets.dtype)
