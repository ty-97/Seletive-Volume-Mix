"""
Frozen BN layers in ResNet
"""
import torch


class FrozenBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(FrozenBatchNorm, self).__init__()
        assert affine
        assert track_running_stats

        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        to_shape = (1, -1) + (1,) * (x.dim() - 2)
        w = self.weight.reshape(to_shape)
        b = self.bias.reshape(to_shape)
        rv = self.running_var.reshape(to_shape)
        rm = self.running_mean.reshape(to_shape)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"

