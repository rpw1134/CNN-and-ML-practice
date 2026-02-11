from .l1 import build as l1_build
from .l2 import build as l2_build
from .elastic_net import build as elastic_net_build

regularization_map = {
    "l1": l1_build,
    "l2": l2_build,
    "elastic_net": elastic_net_build}