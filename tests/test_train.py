from copy import deepcopy

import pytest
import torch

import train
from model import RentalsNet
from prepare import make_loaders
from train import fit_model, run_epoch


def test_network_memorizes_small_batch(prepared_data):
    features = prepared_data.train_dataset.features[:16]
    targets = prepared_data.train_dataset.targets[:16]
    loader = [(features, targets)]

    model = RentalsNet(prepared_data.n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss, _ = run_epoch(model, loader)

    for _ in range(500):
        run_epoch(model, loader, optimizer)

    final_loss, _ = run_epoch(model, loader)

    assert final_loss < initial_loss * 0.01


def test_training_clears_old_gradients(prepared_data):
    clean_model = RentalsNet(prepared_data.n_features)
    dirty_model = deepcopy(clean_model)

    features = prepared_data.train_dataset.features[:16]
    targets = prepared_data.train_dataset.targets[:16]
    loader = [(features, targets)]

    # Old gradients must not affect the next training step.
    for parameter in dirty_model.parameters():
        parameter.grad = torch.full_like(parameter, 1000.0)

    clean_optimizer = torch.optim.SGD(clean_model.parameters(), lr=0.001)
    dirty_optimizer = torch.optim.SGD(dirty_model.parameters(), lr=0.001)

    run_epoch(clean_model, loader, clean_optimizer)
    run_epoch(dirty_model, loader, dirty_optimizer)

    for clean, dirty in zip(
        clean_model.parameters(),
        dirty_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(clean, dirty, rtol=0, atol=0)


def test_validation_leaves_weights_and_gradients_unchanged(prepared_data):
    model = RentalsNet(prepared_data.n_features)
    _, val_loader, _ = make_loaders(prepared_data)

    before = {name: value.clone() for name, value in model.state_dict().items()}

    loss, mae = run_epoch(model, val_loader)

    assert loss >= 0
    assert mae >= 0
    assert not model.training

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])

    assert all(parameter.grad is None for parameter in model.parameters())


def test_epoch_metrics_weight_batches_by_row():
    model = RentalsNet(n_features=1, hidden_sizes=(2, 2))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loader = [
        (torch.zeros(2, 1), torch.tensor([1.0, 3.0])),
        (torch.zeros(1, 1), torch.tensor([5.0])),
    ]

    loss, mae = run_epoch(model, loader)

    assert loss == pytest.approx(35 / 3)
    assert mae == pytest.approx(3.0)


def test_epoch_rejects_mismatched_shapes():
    model = torch.nn.Linear(8, 1)
    loader = [(torch.zeros(4, 8), torch.zeros(4))]

    with pytest.raises(ValueError, match="shapes must match"):
        run_epoch(model, loader)


def test_fit_logs_history_and_restores_best_weights(monkeypatch):
    model = RentalsNet(n_features=8)
    validation_losses = iter([3.0, 1.0, 2.0])
    epoch = 0
    saved_states = []

    def controlled_epoch(model, loader, optimizer=None):
        nonlocal epoch

        if optimizer is not None:
            epoch += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(float(epoch))
            return float(4 - epoch), float(epoch)

        saved_states.append(
            {name: value.clone() for name, value in model.state_dict().items()}
        )
        return next(validation_losses), 0.5

    monkeypatch.setattr(train, "run_epoch", controlled_epoch)

    history, best_epoch = fit_model(model, [], [], epochs=3)

    assert len(history) == 3
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert [row["val_loss"] for row in history] == [3.0, 1.0, 2.0]

    required_keys = {
        "epoch",
        "train_loss",
        "train_mae",
        "val_loss",
        "val_mae",
    }
    assert all(required_keys <= row.keys() for row in history)
    assert best_epoch == 2

    for name, value in model.state_dict().items():
        assert torch.equal(value, saved_states[1][name])
        assert not torch.equal(value, saved_states[2][name])
