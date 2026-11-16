import torch
from torch import nn


class RentalsNet(nn.Module):
    """Predict rentals using two or more hidden layers."""

    def __init__(self, n_features, hidden_sizes=(64, 32)):
        super().__init__()

        if len(hidden_sizes) < 2:
            raise ValueError("Use at least two hidden layers.")

        layers = []
        input_size = n_features

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features).squeeze(-1)


if __name__ == "__main__":
    from prepare import make_loaders, prepare_data

    torch.manual_seed(42)
    data = prepare_data()
    train_loader, _, _ = make_loaders(data)
    features, targets = next(iter(train_loader))

    model = RentalsNet(data.n_features)
    predictions = model(features)

    print(model)
    print("Input shape:", features.shape)
    print("Prediction shape:", predictions.shape)
    print("Target shape:", targets.shape)
    print("Prediction dtype:", predictions.dtype)

    assert predictions.shape == targets.shape
    print("Shape check passed.")
