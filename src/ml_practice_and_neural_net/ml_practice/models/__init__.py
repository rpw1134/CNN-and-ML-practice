from .BaseModel import BaseModel
from .LogisticRegressionModel import LogisticRegressionModel
from .SoftmaxRegressionModel import SoftmaxRegressionModel
from .LinearRegressionModel import LinearRegressionModel

__all__ = ['BaseModel', 'LogisticRegressionModel', 'SoftmaxRegressionModel', 'LinearRegressionModel']

model_map = {
    "linear": LinearRegressionModel,
    "logistic": LogisticRegressionModel,
    "softmax": SoftmaxRegressionModel
}

