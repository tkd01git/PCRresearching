# Pool-size fitting Codex migration

This repository is a Codex/GitHub-friendly version of the original Google Colab workflow for pool-size fitting experiments.

## Main files

- `data_poolsizefitting.py`: data preparation, synthetic source generation, graph/sample export, prior construction.
- `function_poolsizefittiing.py`: PCR/qPCR, pooling matrix, sparse reconstruction, sequential inspection logic.
- `run_poolsize_experiment.py`: command-line runner for Codex and local execution.
- `execution_poolsizefitting_multi_seed.ipynb`: original notebook kept for reference/interactive exploration.

## Install

```bash
python -m pip install -r requirements.txt
```

Codex cloud can automatically install common package-manager dependencies. If needed, set the setup script to:

```bash
python -m pip install -r requirements.txt
```

## Smoke test

```bash
python run_poolsize_experiment.py --seeds 1 --pool-sizes 5 10 --sample-size 300 --n-total 1000 --data-source generate --reuse-generated --no-plots
```

## Standard synthetic-data run

```bash
python run_poolsize_experiment.py   --seeds 1 2 3 4 5   --pool-sizes 1:30   --sample-size 3000   --n-total 10000   --data-source generate   --reuse-generated
```

## Reusing generated population/contact CSVs

The first `--data-source generate` run saves generated data here:

```text
company_openabm_outputs/seed_1/population_all.csv
company_openabm_outputs/seed_1/contacts_all.csv
company_openabm_outputs/seed_2/population_all.csv
company_openabm_outputs/seed_2/contacts_all.csv
...
```

To reuse them, run with either:

```bash
python run_poolsize_experiment.py --data-source generate --reuse-generated ...
```

or:

```bash
python run_poolsize_experiment.py --data-source csv ...
```

## Outputs

Default result files:

```text
results/poolsize_fitting_results.csv
results/poolsize_fitting_summary.csv
results/overview.png
results/summary_by_pool_size.png
```

## Initial migration decisions

1. File names are normalized to match import names.
2. The notebook is retained, but the execution workflow is moved to `run_poolsize_experiment.py`.
3. The default data source is the built-in synthetic source generator, not OpenABM build.
4. Experiment settings are command-line arguments.
5. Validation starts from small smoke tests.
