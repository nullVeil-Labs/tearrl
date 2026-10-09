# Reproducing TEAR-CG/S

These steps are copied unchanged from the original, frozen README.

Use a fresh ignored archive export and a fresh ignored output directory. The
following commands assume those paths do not already exist:

```text
git archive --format=tar --output=artifacts/cgs-source-8568edb.tar 8568edb329e3accdeab5f16c763d42853bef30fd
mkdir artifacts/cgs-source-8568edb
tar -xf artifacts/cgs-source-8568edb.tar -C artifacts/cgs-source-8568edb
python scripts/run_cgs_profiled.py --mode confirmatory --source-root artifacts/cgs-source-8568edb --source-commit 8568edb329e3accdeab5f16c763d42853bef30fd --profile-name env8_torch8_8 --output-dir artifacts/cgs-reproduction-env8-torch8-8 --omp-threads 8 --mkl-threads 8 --openblas-threads 8 --torch-intra-threads 8 --torch-interop-threads 8
python -m pytest -q
python scripts/verify_release_snapshot.py --commit HEAD
```

The profiled runner refuses to overwrite the frozen outputs. The older
`scripts/verify_public_snapshot.py` is a hash-bound historical verifier from the
frozen implementation; `scripts/verify_release_snapshot.py` is the current
release scanner. No model checkpoints are required or published.

