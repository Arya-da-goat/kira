"""Evaluate the original held-out split and export measured curves and samples."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from kira.config import TrainingConfig
from kira.hardware import select_device
from kira.training.checkpoint import load_model, training_status
from kira.training.dataset import read_documents, split_documents, data_fingerprint, TokenDataset
from kira.evaluation.metrics import evaluate
from kira.inference.generate import generate


def write_curves(history, path):
    series = [('train_loss', '#3b82f6'), ('validation_loss', '#10b981')]
    values = [row[key] for key, _ in series for row in history if key in row]
    if not values:
        raise ValueError('No measured losses in checkpoint')
    max_step = max(row['step'] for row in history)
    low, high = min(values), max(values)
    span = max(high - low, 0.01)
    content = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">',
               '<rect width="800" height="400" fill="white"/>',
               '<path d="M60 30V340H770" fill="none" stroke="#111"/>',
               '<text x="60" y="20">Measured cross entropy loss</text>',
               f'<text x="12" y="45">{high:.2f}</text><text x="12" y="340">{low:.2f}</text>',
               f'<text x="350" y="385">Optimizer step (1–{max_step})</text>']
    for index, (key, color) in enumerate(series):
        points = [(60 + 710 * row['step'] / max_step, 340 - 300 * (row[key] - low) / span)
                  for row in history if key in row]
        coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        content.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            content.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>')
        content.append(f'<text x="{80 + index * 240}" y="365" fill="{color}">{key}</text>')
    Path(path).write_text('\n'.join(content + ['</svg>']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/latest.pt')
    parser.add_argument('--data', help='Must match the checkpoint dataset and split')
    parser.add_argument('--output', default='runs/evaluation')
    parser.add_argument('--prompt', action='append', help='Repeat for multiple generation samples')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    config = TrainingConfig(**checkpoint['config']['training'])
    train, validation = split_documents(read_documents(args.data or checkpoint['config']['data']),
                                        config.validation_fraction, config.seed)
    if data_fingerprint(train, validation) != checkpoint['state']['data_fingerprint']:
        raise ValueError('Evaluation data does not match the recorded held-out split')
    loader = DataLoader(TokenDataset(validation, tokenizer, model.config.max_seq_len), batch_size=config.batch_size)
    result = evaluate(model, loader, device)
    result['training_status'] = training_status(checkpoint['state'])
    result['training_step'] = checkpoint['state']['step']
    result['samples'] = [{'prompt': prompt, **generate(model, tokenizer, prompt, seed=config.seed)}
                         for prompt in (args.prompt or ['hello'])]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    write_curves(checkpoint['state']['history'], output / 'loss.svg')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
