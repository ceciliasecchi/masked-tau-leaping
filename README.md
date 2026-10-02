# Schedule optimization for tau-leaping in masked discrete diffusion

Code accompanying the paper's numerical experiments.

## Files

| File | Description |
| --- | --- |
| `optimal_schedules.py` | Functions for mixture targets, information-profile estimation, dependence densities, numerical schedule optimization, and factorization-error evaluation. |
| `Mixtures-examples.ipynb` | Mixture-of-products experiments comparing linear and optimized schedules using estimated profiles and direct Monte Carlo evaluation. |
| `Markov-exchaneable-examples.ipynb` | Stationary Markov-chain and Beta–Bernoulli examples, evaluated using explicit dependence-density formulas and numerical integration. |


## Experiment settings

**Mixtures:** five components, vocabulary size 10, Dirichlet concentration 0.3, seed 7, and sequence lengths 8–256 (powers of two), with $K=\log_2 N$. The targets are nested prefixes of one fixed mixture. Each profile uses 40,000 draws; isotonic regression is applied after pooling the raw estimates. Each schedule is evaluated with 100 independent Monte Carlo estimates of 4,000 draws each.

**Markov and exchangeable examples:** a 10-state lazy random walk and a Beta(1,1)–Bernoulli mixture, with $K=1+\lceil\log_2 N\rceil$. Dependence densities and errors are evaluated without Monte Carlo sampling of target sequences.

## Outputs

The notebooks save the manuscript figures in PNG format:

- `mixture_experiment_four_panels`
- `factorization_schedules`

Files are written to the notebook's working directory.
