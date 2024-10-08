import torch
from string import Template
from collections import namedtuple
try:
    import cupy
except ImportError:
    cupy = None

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


#@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    # kernel_code = cupy.cuda.compile_with_cache(code) # cupy version < 10x
    # return kernel_code.get_function(kernel_name) # cupy version < 10x
    kernel_code = cupy.RawKernel(code, kernel_name)
    return kernel_code
