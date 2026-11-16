import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from prepare import NUMERIC, make_loaders


def test_dataset_length_shapes_and_dtypes(prepared_data):
    dataset = prepared_data.train_dataset
    features, target = dataset[0]

    assert len(dataset) == 3600
    assert features.shape == (prepared_data.n_features,)
    assert target.shape == ()
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


def test_splits_have_expected_sizes_and_do_not_overlap(prepared_data):
    train = set(prepared_data.train_frame.index)
    val = set(prepared_data.val_frame.index)
    test = set(prepared_data.test_frame.index)

    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(6000))


def test_scaler_is_fitted_on_training_rows_only(prepared_data):
    scaler = prepared_data.preprocessor.named_transformers_["numeric"]

    expected_mean = prepared_data.train_frame[NUMERIC].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_mean)
    assert scaler.n_samples_seen_ == 3600

    train_scaled = scaler.transform(prepared_data.train_frame[NUMERIC])
    val_scaled = scaler.transform(prepared_data.val_frame[NUMERIC])

    np.testing.assert_allclose(train_scaled.mean(axis=0), 0.0, atol=1e-12)
    assert not np.allclose(val_scaled.mean(axis=0), 0.0, atol=1e-3)


def test_training_loader_shuffles_repeatably(prepared_data):
    first, val_loader, test_loader = make_loaders(prepared_data, seed=42)
    second, _, _ = make_loaders(prepared_data, seed=42)

    first_order = list(first.sampler)
    second_order = list(second.sampler)

    assert isinstance(first.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    assert first_order == second_order
    assert first_order != list(range(len(prepared_data.train_dataset)))
    assert sorted(first_order) == list(range(len(prepared_data.train_dataset)))
