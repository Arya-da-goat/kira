"""Train BPE on the training documents only, using the training split settings."""
import argparse
import json
from pathlib import Path
from kira.config import TrainingConfig
from kira.tokenizer.tokenizer import Tokenizer
from kira.training.dataset import read_documents, split_documents, data_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='configs/tiny.json')
    parser.add_argument('--output', default='checkpoints/tokenizer.json')
    parser.add_argument('--vocab-size', type=int, default=None)
    args = parser.parse_args()
    train_tokenizer(args.data, args.config, args.output, args.vocab_size)


def train_tokenizer(data, config_path, output, vocab_size=None):
    """Fit vocabulary only on the reproducible training split and save provenance."""
    output = str(output)
    config = json.loads(Path(config_path).read_text())
    training = TrainingConfig(**config['training'])
    train, validation = split_documents(read_documents(data), training.validation_fraction, training.seed)
    tokenizer = Tokenizer.train(train, vocab_size or config['model']['vocab_size'])
    tokenizer.save(output)
    # Provenance is kept separately; the vocabulary format stays reusable.
    Path(output + '.provenance.json').write_text(json.dumps({
        'data_fingerprint': data_fingerprint(train, validation),
        'train_documents': len(train), 'validation_documents': len(validation),
        'tokenizer_fingerprint': tokenizer.fingerprint,
    }, indent=2))
    print(f'Saved {tokenizer.vocab_size} tokens to {output}; fitted on {len(train)} training documents only.')


if __name__ == '__main__':
    main()
