from hyperbolic_tan import build as build_hyperbolic_tan
from sigmoid import build as build_sigmoid
from relu import build as build_relu
from leaky_relu import build as build_leaky_relu

activation_map = {
    'hyperbolic_tan': build_hyperbolic_tan,
    'sigmoid': build_sigmoid,
    'relu': build_relu,
    'leaky_relu': build_leaky_relu
}

__all__ = ['activation_map', 'build_hyperbolic_tan', 'build_sigmoid', 'build_relu', 'build_leaky_relu']
