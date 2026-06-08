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

## OpenABM sample generation

Generate reusable samples from real OpenABM-Covid19 runs:

```bash
python run_openabm_sample_generation.py --seeds 1:10 --sample-size 3000 --reuse-converted --reuse-samples --delete-raw-after-convert
```

The script clones/builds OpenABM-Covid19 under `openabm_work/`, builds a local
GSL dependency if needed, runs OpenABM once per seed, converts raw output into
`population_all.csv` / `contacts_all.csv`, and exports sample files under:

```text
openabm_sample_outputs/seed_1/samples/company_n3000_maxpos5pct_work/
openabm_sample_outputs/seed_2/samples/company_n3000_maxpos5pct_work/
...
```

OpenABM raw output and full converted source CSVs are intentionally ignored by
git because they are large. The extracted sample files are small and can be
committed to GitHub.

To extract multiple non-overlapping samples from each OpenABM run, add
`--samples-per-seed`. For example:

```bash
python run_openabm_sample_generation.py --seeds 1:10 --sample-size 3000 --samples-per-seed 5 --reuse-converted --reuse-samples --delete-raw-after-convert
```

This creates directories such as:

```text
openabm_sample_outputs/seed_1/samples/company_n3000_sample01_maxpos5pct_work/
openabm_sample_outputs/seed_1/samples/company_n3000_sample02_maxpos5pct_work/
...
```

By default, samples that do not meet the diagnostic thresholds are not saved.
The extractor retries with different extraction RNG seeds up to
`--max-sample-attempts`. Use `--allow-best-effort-samples` only when you want to
save the best available sample even if a diagnostic threshold is missed.

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
