from cce import build as build_cce
from mse import build as build_mse
from ce import build as build_ce

loss_map = {"mse": build_mse,
            "cce": build_cce,
            "ce": build_ce}

__all__ = ["loss_map", "build_cce", "build_mse", "build_ce"]