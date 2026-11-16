import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from explore import plot_rentals
from model import RentalsNet
from prepare import make_loaders, prepare_data


def check_small_batch():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    data = prepare_data()
    features = data.train_dataset.features[:16]
    targets = data.train_dataset.targets[:16]

    model = RentalsNet(data.n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()

    with torch.no_grad():
        initial_loss = loss_fn(model(features), targets).item()

    for step in range(1, 1001):
        optimizer.zero_grad()
        predictions = model(features)

        if predictions.shape != targets.shape:
            raise ValueError("Prediction and target shapes must match.")

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step:4d} | MSE: {loss.item():.6f}")

    with torch.no_grad():
        final_loss = loss_fn(model(features), targets).item()

    print(f"\nInitial MSE: {initial_loss:.6f}")
    print(f"Final MSE:   {final_loss:.6f}")
    print(f"Remaining loss: {100 * final_loss / initial_loss:.4f}%")

    assert final_loss < initial_loss * 0.01, (
        "The network did not reduce the loss below 1% of its initial value."
    )
    print("Small-batch learning check passed.")


def run_epoch(model, loader, optimizer=None):
    """Train with an optimizer; otherwise measure without changing weights."""
    training = optimizer is not None
    model.train(training)

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_rows = 0
    loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(training):
        for features, targets in loader:
            if training:
                optimizer.zero_grad()

            predictions = model(features)

            if predictions.shape != targets.shape:
                raise ValueError("Prediction and target shapes must match.")

            loss = loss_fn(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_rows = targets.shape[0]
            total_squared_error += loss.item() * batch_rows
            total_absolute_error += (predictions - targets).abs().sum().item()
            total_rows += batch_rows

    return (
        total_squared_error / total_rows,
        total_absolute_error / total_rows,
    )


def fit_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    learning_rate=0.001,
):
    """Log every epoch and restore the lowest-validation-loss weights."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_loss = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:3d} | "
            f"Train MSE: {train_loss:.3f} | "
            f"Val MSE: {val_loss:.3f} | "
            f"Val MAE: {val_mae:.3f}"
        )

    model.load_state_dict(best_state)
    return history, best_epoch


def save_learning_curve(history, best_epoch, output_path):
    history_frame = pd.DataFrame(history)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        history_frame["epoch"],
        history_frame["train_loss"],
        label="Training loss",
    )
    ax.plot(
        history_frame["epoch"],
        history_frame["val_loss"],
        label="Validation loss",
    )
    ax.axvline(
        best_epoch,
        color="black",
        linestyle="--",
        label=f"Best epoch: {best_epoch}",
    )
    ax.set(
        title="Bike rental learning curves",
        xlabel="Epoch",
        ylabel="Mean squared error (rentals squared)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_baselines(data):
    train_x = data.train_dataset.features.numpy()
    train_y = data.train_dataset.targets.numpy()

    linear = LinearRegression()
    linear.fit(train_x, train_y)
    training_mean = float(train_y.mean())

    results = {}

    for name, dataset in [
        ("validation", data.val_dataset),
        ("test", data.test_dataset),
    ]:
        features = dataset.features.numpy()
        targets = dataset.targets.numpy()

        predictions = {
            "training_mean": np.full(targets.shape, training_mean, dtype=np.float32),
            "linear_regression": linear.predict(features),
        }

        for model_name, predicted in predictions.items():
            results.setdefault(model_name, {})[name] = {
                "mse": float(mean_squared_error(targets, predicted)),
                "mae": float(mean_absolute_error(targets, predicted)),
            }

    return results


if __name__ == "__main__":
    torch.set_num_threads(1)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    data = prepare_data()

    configurations = [
        {
            "name": "main",
            "hidden_sizes": (64, 32),
            "learning_rate": 0.001,
        },
        {
            "name": "slow_learning",
            "hidden_sizes": (64, 32),
            "learning_rate": 0.00001,
        },
        {
            "name": "wide_network",
            "hidden_sizes": (256, 128),
            "learning_rate": 0.001,
        },
    ]

    selected_model = None
    selected_run = None
    selected_epoch = None
    selected_val_loss = float("inf")

    plot_rentals(data.train_frame, reports)

    experiment_rows = []

    for config in configurations:
        print(f"\n--- Run: {config['name']} ---")

        # Reset both model and loader randomness for each run.
        torch.manual_seed(42)
        train_loader, val_loader, _ = make_loaders(data, batch_size=64, seed=42)
        model = RentalsNet(
            data.n_features,
            hidden_sizes=config["hidden_sizes"],
        )

        history, best_epoch = fit_model(
            model,
            train_loader,
            val_loader,
            epochs=200,
            learning_rate=config["learning_rate"],
        )

        best_run_loss = history[best_epoch - 1]["val_loss"]

        if best_run_loss < selected_val_loss:
            selected_val_loss = best_run_loss
            selected_model = model
            selected_run = config["name"]
            selected_epoch = best_epoch

        history_frame = pd.DataFrame(history)
        best_row = history[best_epoch - 1]
        final_row = history[-1]

        if config["name"] == "main":
            history_path = reports / "history.csv"
            curve_path = reports / "learning_curves.png"
        else:
            history_path = reports / f"history_{config['name']}.csv"
            curve_path = reports / f"learning_curves_{config['name']}.png"

        history_frame.to_csv(history_path, index=False)
        save_learning_curve(history, best_epoch, curve_path)

        experiment_rows.append(
            {
                "run": config["name"],
                "hidden_sizes": str(config["hidden_sizes"]),
                "learning_rate": config["learning_rate"],
                "batch_size": 64,
                "epochs": 200,
                "best_epoch": best_epoch,
                "best_val_loss": best_row["val_loss"],
                "val_mae_at_best_loss": best_row["val_mae"],
                "minimum_val_mae": history_frame["val_mae"].min(),
                "final_train_loss": final_row["train_loss"],
                "final_val_loss": final_row["val_loss"],
            }
        )

    experiments = pd.DataFrame(experiment_rows)
    experiments.to_csv(reports / "experiments.csv", index=False)

    print("\n--- Experiment comparison ---")
    print(experiments.to_string(index=False))
    print("\nSaved experiment table, histories, and curves.")

    # Model selection is complete before evaluating the test rows.
    _, val_loader, test_loader = make_loaders(data)

    val_mse, val_mae = run_epoch(selected_model, val_loader)
    test_mse, test_mae = run_epoch(selected_model, test_loader)

    metrics = evaluate_baselines(data)
    metrics["network"] = {
        "selected_run": selected_run,
        "best_epoch": selected_epoch,
        "validation": {"mse": val_mse, "mae": val_mae},
        "test": {"mse": test_mse, "mae": test_mae},
    }

    (reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSelected network: {selected_run}")
    print(f"Selected epoch: {selected_epoch}")
    print("\nMAE comparison (rentals)")

    for name in ["training_mean", "linear_regression", "network"]:
        result = metrics[name]
        print(
            f"{name:20s} | "
            f"Validation: {result['validation']['mae']:.3f} | "
            f"Test: {result['test']['mae']:.3f}"
        )
