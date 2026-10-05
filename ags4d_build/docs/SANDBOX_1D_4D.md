# AGS-Sci Dimension-Agnostic Experiment Sandbox

The experiment sandbox supports **1D, 2D, 3D and 4D** experiments through a common admission contract. AGS can submit a dimension-aware `ExperimentSpec` without coupling the core to a particular numerical solver.

## Admission contract

Each experiment declares:

- `dimension`: 1–4
- `shape`: one extent per dimension
- `fields`: number of simultaneously resident numerical fields
- `dtype_bytes`: element size
- `steps`: requested numerical steps
- `runtime_s`: estimated runtime

The admission gate checks cell count, fields, estimated working memory, step count and runtime before code is executed.

### Default exploratory budgets

| Dimension | Maximum cells | Maximum fields | Estimated memory | Runtime | Steps |
|---|---:|---:|---:|---:|---:|
| 1D | 10,000,000 | 16 | 512 MB | 30 s | 100,000 |
| 2D | 4,000,000 | 16 | 512 MB | 30 s | 100,000 |
| 3D | 262,144 | 16 | 512 MB | 30 s | 100,000 |
| 4D | 65,536 | 16 | 512 MB | 30 s | 100,000 |

These are **admission budgets, not scientific certification limits**. Validation experiments can use a separate policy later.

## Ephemeral state

Numerical arrays, temporary checkpoints and generated files remain inside the worker's temporary directory. They are not persisted as AGS knowledge. Only the execution result and explicitly returned artifacts cross the sandbox boundary.

## Example

```python
from ags_sci.experiment import ExecutionSandbox, ExperimentSpec

spec = ExperimentSpec(
    dimension=3,
    shape=(32, 32, 32),
    fields=4,
    dtype_bytes=4,
    steps=1000,
    runtime_s=10,
)

with ExecutionSandbox(timeout_seconds=12) as sandbox:
    result = sandbox.execute_experiment(spec, source_code)
```

## Design rule

The sandbox is an execution substrate, not AGS memory. AGS may learn useful terms or relationships from an experiment, but raw experiment state should remain disposable.
