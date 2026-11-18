---
tocdepth: 3
---

```{include} Dialects/QAssertDialect.md

```

## Use with QCO

No pass evaluates, lowers, or removes assertions yet. Assertions implement the
QCO `UnitaryOpInterface` as identities, like `qco.barrier`, but unlike it they
report no unitary matrix. QCO therefore follows their qubits through `qco.if`,
loops, and functions, so assertions can be used inside them. Optimization passes
keep assertions: passes that need a matrix skip them, and their read effect
keeps `remove-dead-gates` from removing them even directly before a sink or
reset.
