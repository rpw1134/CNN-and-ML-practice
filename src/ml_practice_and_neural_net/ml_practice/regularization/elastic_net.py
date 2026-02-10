def build(l1, l2):
    from .l1 import build as build_l1
    from .l2 import build as build_l2

    l1_loss, l1_grad = build_l1(l1)
    l2_loss, l2_grad = build_l2(l2)

    def loss(parameters):
        return l1_loss(parameters) + l2_loss(parameters)

    def gradient(parameters):
        return l1_grad(parameters) + l2_grad(parameters)

    return loss, gradient