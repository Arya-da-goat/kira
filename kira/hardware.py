"""Device selection and explicitly supported autocast modes."""
from contextlib import nullcontext
import torch


def select_device(requested: str = 'auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if requested not in ('cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda, or mps')
    device = torch.device(requested)
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable')
    return device


def select_precision(device, requested='auto'):
    if requested == 'auto':
        return ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device.type == 'cuda' else 'fp32'
    if requested not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be auto, fp32, fp16, or bf16')
    if requested != 'fp32' and device.type != 'cuda':
        raise ValueError('Mixed precision is supported on CUDA only; use fp32 here')
    if requested == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support BF16')
    return requested


def autocast_context(device, precision):
    return nullcontext() if precision == 'fp32' else torch.autocast(device_type=device.type, dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[precision])


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def describe(device, precision):
    result = {'device': str(device), 'precision': precision}
    if device.type == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(device)
    return result
