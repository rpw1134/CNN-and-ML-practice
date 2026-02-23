from dataclasses import dataclass

@dataclass
class NNLayerParameters:
    width: int = 16
    input_dim: int = 16
    activation: str = "relu"    # Options could include "relu", "sigmoid", "tanh", "leaky_relu", etc. for different activation functions
    weight_init: str = "xavier" # Options could include "xavier", "he", "random", etc. for different weight initialization strategies
    sparsity: float = 1.0       # Fraction of connections to keep (1.0 = dense, 0.5 = 50% connected)
