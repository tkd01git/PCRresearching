# AGENTS.md

## Project goal
Migrate the pool-size fitting research workflow from Google Colab to a Codex/GitHub-friendly Python project.

## Default validation commands
Run a small smoke test before proposing changes:

```bash
python run_poolsize_experiment.py --seeds 1 --pool-sizes 5 10 --sample-size 300 --n-total 1000 --data-source generate --reuse-generated --no-plots
```

For a slightly larger check:

```bash
python run_poolsize_experiment.py --seeds 1 2 --pool-sizes 1 5 10 --sample-size 300 --n-total 1000 --data-source generate --reuse-generated
```

## Notes
- Keep `data_poolsizefitting.py` and `function_poolsizefittiing.py` importable from the repository root.
- Generated CSVs are stored under `company_openabm_outputs/seed_<seed>/`.
- Results are stored under `results/` and should normally not be committed, except for tiny examples if explicitly requested.
- Do not require OpenABM build for the default workflow. The default must work with the synthetic source generator.
