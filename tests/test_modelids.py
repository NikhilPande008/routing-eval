import pytest

from routing_eval.modelids import normalize_model_id, split_models


def test_normalize_adds_the_fireworks_prefix_to_a_bare_name():
    assert normalize_model_id("minimax-m3") == "accounts/fireworks/models/minimax-m3"


def test_normalize_is_idempotent_on_an_already_full_path():
    full = "accounts/fireworks/models/minimax-m3"
    assert normalize_model_id(full) == full


def test_split_models_normalizes_every_entry():
    assert split_models("minimax-m3,kimi-k2p7-code") == [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]


def test_split_models_leaves_already_full_paths_untouched():
    assert split_models("accounts/fireworks/models/minimax-m3, kimi-k2p7-code") == [
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/kimi-k2p7-code",
    ]


def test_split_models_rejects_empty():
    with pytest.raises(ValueError):
        split_models("   ")
