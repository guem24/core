---
tocdepth: 3
---

```{include} Dialects/QAssertDialect.md

```

## Use with QCO

No pass evaluates, lowers, or removes assertions yet. Optimization passes keep
them because each assertion consumes and returns its qubits.

Assertions implement the QCO `QubitPassThroughOpInterface`. QCO therefore
follows their qubits through structured control flow and functions, so
assertions can be used inside both.
