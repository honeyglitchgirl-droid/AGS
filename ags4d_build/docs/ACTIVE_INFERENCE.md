# AGS Active-Inference Research Architecture

The active research stack is organized by semantic responsibility rather than development version.

## Flow

```text
LLM proposal
   -> strict Pydantic causal schema
   -> adversarial triad (proposer / skeptic / methodologist)
   -> symbolic + dimensional verification
   -> causal intervention / experiment protocol
   -> BOED / Expected Free Energy policy ranking
   -> Causal MCTS hypothesis-tree search
   -> sandboxed dry-lab execution
   -> deterministic likelihood calculation
   -> Bayes-factor evidence update
   -> falsification / promotion decision
```

## Trust boundaries

- `ags_core`: pure contracts and deterministic mathematical semantics.
- `ags_epistemic`: Bayesian evidence and active-inference objectives.
- `ags_experiment`: symbolic verification and process-isolated execution.
- `ags_search`: model-tree exploration using UCT.
- `adapters`: untrusted external AI and simulation boundaries.
- `discovery`: orchestration only; it does not become the statistical authority.

LLMs may generate candidate mechanisms, but numerical evidence, graph validity,
likelihoods, and promotion decisions remain deterministic system responsibilities.

## Security note

The local sandbox is a defense-in-depth process boundary with timeout and best-effort
resource limits. Production deployment should add OS/container isolation such as
WASI or a seccomp-constrained container. The sandbox is not presented as a complete
host-security boundary by itself.
