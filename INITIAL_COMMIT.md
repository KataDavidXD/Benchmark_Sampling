# Initial Commit: WTB Sampling Component

**feat: implement core sampling engine and stratification modules**

This commit introduces the comprehensive sampling subsystem for the WTB RAG benchmark optimization, featuring stratified, Neyman, and Metropolis-Hastings (MH) adaptive sampling capabilities. 

### Core Components
*   **`SamplingEngine`**: Unified facade API orchestrating data loading, stratification, sampling, and sequential estimation.
*   **Adapters**: Implemented `UltraDomainAdapter` and `FreshWikiAdapter` to map raw dataset rows into standard `BenchmarkItem` formats (payload/target/metadata).
*   **Stratification**: Logic to partition items by primary axes (e.g., domain, quality bucket) and secondary axes (context length), including auto-collapsing of small strata.
*   **Samplers**:
    *   `StratifiedSampler`: Supports both proportional allocation and variance-optimized Neyman allocation (via two-phase pilot).
    *   `MetropolisHastingsSampler`: MCMC-based adaptive allocation that searches for the minimum-variance allocation vector.
*   **Estimator & Stopping**: `SequentialEstimator` for online stratified mean/variance and confidence interval (CI) updates, coupled with a `BudgetController` and configurable early stopping rules.
*   **Comparison**: Added `paired_compare` (paired t-test) and `should_eliminate` (non-overlapping CI) to evaluate RAG configurations on shared item subsets.

### Infrastructure & Documentation
*   **Types**: Serializable data models (`BenchmarkItem`, `SamplingState`, `ItemRealization`, etc.) supporting WTB checkpointing and rollback.
*   **Diagnostics**: Metrics including Effective Sample Size (ESS), variance reduction, and MH acceptance/energy traces.
*   **Tests**: Full suite of synthetic and real-data integration tests (`pytest`).
*   **Examples**: `main.py` providing runnable examples of Neyman, Proportional, and MH sampling with detailed output interpretation.
*   **Docs**: Comprehensive `README.md` and `docs/SAMPLING_ARCHITECTURE.md` detailing the design, SOLID/ACID principles, and data flow.
*   **Config**: Initial `pyproject.toml` and `.gitignore`.