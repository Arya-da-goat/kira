"""Isolated training process used by the web job controller."""
import argparse
import json
from pathlib import Path

import torch
from kira.tokenizer.train import train_tokenizer
from kira.training.train import train


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    directory = Path(args.run_dir)
    request = json.loads((directory / 'job.json').read_text())
    phase = directory / 'phase.json'
    torch.set_num_threads(args.threads)
    try:
        write_json(phase, {'phase': 'Training tokenizer on training split'})
        data = directory / ('dataset.' + request['format'])
        config = directory / 'config.json'
        tokenizer = directory / 'tokenizer.json'
        train_tokenizer(data, config, tokenizer)
        write_json(phase, {'phase': 'Training Transformer from random initialization'})
        train(config, data, tokenizer, directory, device_name=args.device)
        write_json(phase, {'phase': 'Training finished; quality unverified'})
    except Exception as exc:
        write_json(phase, {'phase': 'Failed', 'error': str(exc)[:500]})
        raise


if __name__ == '__main__':
    main()
