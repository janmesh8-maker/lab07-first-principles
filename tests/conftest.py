import pytest
import torch

from prepare import prepare_data


@pytest.fixture(scope="session")
def prepared_data():
    return prepare_data(seed=42)


@pytest.fixture(autouse=True)
def reset_torch_seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)
