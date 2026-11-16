import pytest
import torch
from torch import nn

from model import RentalsNet
from prepare import make_loaders
from train import run_epoch


@pytest.mark.parametrize(
    "hidden_sizes",
    [(8, 4), (64, 32), (256, 128)],
)
def test_network_returns_one_value_per_row(hidden_sizes):
    model = RentalsNet(n_features=8, hidden_sizes=hidden_sizes)

    for batch_size in [1, 7, 64]:
        features = torch.randn(batch_size, 8)
        predictions = model(features)

        assert predictions.shape == (batch_size,)
        assert predictions.dtype == torch.float32
        assert torch.isfinite(predictions).all()


def test_optimizer_step_changes_every_layer(prepared_data):
    model = RentalsNet(prepared_data.n_features)
    train_loader, _, _ = make_loaders(prepared_data, batch_size=64)

    # Use one batch through the actual epoch function.
    batch = next(iter(train_loader))
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith("weight")
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    run_epoch(model, [batch], optimizer)

    for name, parameter in model.named_parameters():
        if name in before:
            assert not torch.equal(before[name], parameter), (
                f"Layer weights did not change: {name}"
            )


def test_network_has_linear_output():
    model = RentalsNet(n_features=8)

    assert isinstance(model.network[-1], nn.Linear)
    assert model.network[-1].out_features == 1
    assert sum(isinstance(layer, nn.ReLU) for layer in model.network) >= 2
