from .initializers import normal, he_normal, xavier_normal, uniform, he_uniform, xavier_uniform


initializers_map = {"normal": normal,
                    "he_normal": he_normal,
                    "xavier_normal": xavier_normal,
                    "uniform": uniform,
                    "he_uniform": he_uniform,
                    "xavier_uniform": xavier_uniform}

__all__ = ["initializers_map", "normal", "he_normal", "xavier_normal", "uniform", "he_uniform", "xavier_uniform"]