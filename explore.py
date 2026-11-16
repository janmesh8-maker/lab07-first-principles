from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from bikes_data import make_rentals


def plot_rentals(frame, output_dir="reports"):
    """Plot the training rows so the test data stay untouched."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        frame["hour"],
        frame["rentals"],
        alpha=0.12,
        s=12,
        label="Station-hours",
    )

    hourly_mean = frame.groupby("hour")["rentals"].mean()
    ax.plot(
        hourly_mean.index,
        hourly_mean.values,
        color="red",
        marker="o",
        label="Mean rentals",
    )

    ax.set(
        title="Bike rentals by hour — training data",
        xlabel="Hour of day",
        ylabel="Bikes rented",
        xticks=range(0, 24, 2),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rentals_by_hour.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    frame = make_rentals()
    train_frame, _ = train_test_split(
        frame,
        test_size=0.4,
        random_state=42,
    )
    plot_rentals(train_frame)
    print("Saved reports/rentals_by_hour.png")
