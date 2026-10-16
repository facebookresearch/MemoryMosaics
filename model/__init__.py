# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""Model families, selected by the top-level `model_type` of a config:

    mosaics      Memory Mosaics, MMv2, MM+, two-stage MM+, Transformer+ (model/mosaics)
    transformer  Transformer, Hybrid SWA, Hybrid DeltaNet                 (model/transformer)
    rnn          DeltaNet                                                 (model/rnn)

Layers they share (attention, DeltaNet, masks, positions, ...) are in model/common.
"""

from model.mosaics.config import MosaicsConfig
from model.mosaics.model import MemoryMosaics
from model.rnn.config import RNNConfig
from model.rnn.model import RNN
from model.transformer.config import TransformerConfig
from model.transformer.model import Transformer

# model_type -> (config class, model class)
FAMILIES = {
    "mosaics": (MosaicsConfig, MemoryMosaics),
    "transformer": (TransformerConfig, Transformer),
    "rnn": (RNNConfig, RNN),
}


def build_model(model_type: str, config):
    assert model_type in FAMILIES, f"model_type={model_type!r} not in {sorted(FAMILIES)}"
    return FAMILIES[model_type][1](config)


def config_from_dict(model_type: str, values: dict):
    """The model config of `model_type` from a dict (e.g. params.json), ignoring nothing:
    an unknown key is an error."""
    from dataclasses import fields
    cls = FAMILIES[model_type][0]
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    assert not unknown, f"unknown {model_type} model settings {unknown}"
    return cls(**values)
