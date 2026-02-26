from ..nn_loss.ce import loss as ce_loss, gradient as ce_gradient
from ..nn_loss.cce import loss as cce_loss, gradient as cce_gradient
from ..nn_loss.mse import loss as mse_loss, gradient as mse_gradient

nn_loss_map = {"mse": (mse_loss, mse_gradient), "ce": (ce_loss, ce_gradient), "cce": (cce_loss, cce_gradient)}

__all__ = ["nn_loss_map", "ce_loss", "ce_gradient", "cce_loss", "cce_gradient", "mse_loss", "mse_gradient"]