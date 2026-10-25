---
tocdepth: 3
---

```{include} Dialects/QAssertDialect.md

```

## Use with QCO

QAssert operations act on QCO qubits. QCO does not depend on QAssert. The
operations do not implement the unitary interface, so QCO rejects them inside
`qco.ctrl`, `qco.inv`, and `qco.pow` bodies.

No pass evaluates, lowers, or removes assertions yet. Optimization passes keep
them because each assertion consumes and returns its qubits.
