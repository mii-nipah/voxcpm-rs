# Weight-loading fixture

`tiny-state-dict.pth` is a synthetic, uncompressed PyTorch ZIP state dictionary.
It contains `state_dict.linear.weight`: an F32 tensor with shape `[2, 3]` and
values `[[1, 2, 3], [4, 5, 6]]`. No model or user data is included.

The fixture exercises PyTorch's storage reference, tensor rebuild metadata,
container-prefix stripping, and lazy backing-storage lifetime.
