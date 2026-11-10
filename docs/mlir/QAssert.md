---
tocdepth: 3
---

```{include} Dialects/QAssertDialect.md

```

## Use with QCO

No pass evaluates, lowers, or removes assertions yet. Optimization passes keep
them because each assertion consumes and returns its qubits.

Assertions implement the QCO `QubitPassThroughOpInterface`. QCO therefore
follows their qubits through structured control flow, functions, placement and
routing, and target-native synthesis, so assertions can be used in all of them.
