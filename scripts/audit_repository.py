"""Check the current files that `git add -A` would include, without staging them.

This is a conservative hygiene/secret-pattern check, not a guarantee that every
possible secret is detected. Diagnostics never print matching secret values.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


FORBIDDEN_DIRECTORIES = {
    '.venv', 'venv', 'env', 'node_modules', '__pycache__', '.pytest_cache',
    '.mypy_cache', '.ruff_cache', '.idea', '.vscode', 'dist', 'build',
    '.cache', '.tox', '.nox', 'htmlcov', 'coverage',
}
FORBIDDEN_SUFFIXES = {
    '.pt', '.pth', '.safetensors', '.ckpt', '.onnx', '.pyc', '.pyo', '.log',
    '.tmp', '.temp', '.bak', '.pem', '.key', '.p12', '.pfx', '.swp', '.swo',
}
SECRET_PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'provider token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,})\b'),
    'credential in URL': re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    'literal credential': re.compile(r'''(?i)\b(?:api[_-]?key|password|client[_-]?secret|access[_-]?token|auth[_-]?token)\b["']?\s*[:=]\s*["'][^"'\s]{8,}["']'''),
    'developer machine path': re.compile(r'(?:/(?:home|Users)/[^/\s]+/|[A-Za-z]:\\Users\\)'),
}


def candidates(root: Path) -> list[Path]:
    result = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root
    )
    names = sorted(set(result.decode('utf-8').split('\0')) - {''})
    return [root / name for name in names if (root / name).exists() or (root / name).is_symlink()]


def audit(root: Path) -> tuple[list[str], int, int]:
    issues, total = [], 0
    files = candidates(root)
    for file in files:
        relative = file.relative_to(root)
        name = relative.as_posix()
        if file.is_symlink() or not file.is_file():
            issues.append(f'{name}: only regular source files are allowed')
            continue
        parts = relative.parts
        forbidden = (
            any(part in FORBIDDEN_DIRECTORIES or part.endswith('.egg-info') for part in parts)
            or file.suffix.lower() in FORBIDDEN_SUFFIXES
            or (file.name.startswith('.env') and file.name != '.env.example')
            or file.name in {'.DS_Store', 'Thumbs.db', '.coverage'}
            or file.name.startswith(('credentials.', 'secrets.'))
            or parts[0] in {'runs', 'output', 'outputs', 'tmp'}
            or (parts[0] == 'checkpoints' and name != 'checkpoints/README.md')
            or (parts[0] == 'data' and name not in {'data/README.md', 'data/example/tiny.txt'})
        )
        if forbidden:
            issues.append(f'{name}: generated, private or local-only file')
        size = file.stat().st_size
        total += size
        limit = 64 * 1024 if name == 'data/example/tiny.txt' else 1024 * 1024
        if size > limit:
            issues.append(f'{name}: exceeds source-file size limit ({size} bytes)')
            continue
        try:
            content = file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            issues.append(f'{name}: binary content needs explicit review')
            continue
        for number, line in enumerate(content.splitlines(), 1):
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    issues.append(f'{name}:{number}: potential {label}; inspect locally')
    return issues, len(files), total


def main() -> int:
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    issues, count, size = audit(root)
    if issues:
        print('\n'.join(issues), file=sys.stderr)
        return 1
    print(f'Audit passed: {count} Git candidate files, {size:,} bytes; no forbidden artifacts or matching secret patterns.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
