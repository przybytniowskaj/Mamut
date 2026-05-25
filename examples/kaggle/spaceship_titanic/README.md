# MAMUT Spaceship Titanic Notebook

This folder contains the source for the Kaggle Code notebook
`MAMUT: Auditable Spaceship Titanic Baseline`. It demonstrates MAMUT as a
transparent competition baseline: target-free feature construction,
passenger-group-aware evaluation, baseline evidence, and controlled submission
file generation.

The notebook does not claim to reproduce the repository's recorded `0.80617`
public score exactly. That score came from a recorded post-leaderboard campaign
in `scripts/benchmark_kaggle.py`; see the documentation for its protocol and
limitations.

After `mamut==0.3.0` is available on PyPI, publish a private validation run
with:

```sh
kaggle kernels push -p examples/kaggle/spaceship_titanic
```

Only make the notebook public after its Kaggle execution completes
successfully and produces `submission.csv`.
