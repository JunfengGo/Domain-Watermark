"""Setup constants, ymmv."""

NORMALIZE = True  # Normalize all datasets

PIN_MEMORY = True
NON_BLOCKING = True
BENCHMARK = True
MAX_THREADING = 40
SHARING_STRATEGY = 'file_descriptor'  # file_system or file_descriptor

DISTRIBUTED_BACKEND = 'gloo'  # nccl would be faster, but require gpu-transfers for indexing and stuff

cifar10_mean = [0.485, 0.456, 0.406]
cifar10_std = [0.229, 0.224, 0.225]
cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
tiny_imagenet_mean = [0.485, 0.456, 0.406]
tiny_imagenet_std = [0.229, 0.224, 0.225]
