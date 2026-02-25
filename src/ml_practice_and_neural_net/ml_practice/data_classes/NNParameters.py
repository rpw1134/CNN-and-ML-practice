from dataclasses import dataclass

@dataclass
class NNParameters:
    loss: str = "mse"