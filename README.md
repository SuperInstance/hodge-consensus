# hodge-consensus

[![crates.io](https://img.shields.io/crates/v/hodge-consensus.svg)](https://crates.io/crates/hodge-consensus)
[![docs.rs](https://docs.rs/hodge-consensus/badge.svg)](https://docs.rs/hodge-consensus)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

Three agents rate items A, B, C: Agent 1 says A > B > C. Agent 2 says B > C > A. Agent 3 says C > A > B. Each agent is internally consistent, but collectively they're going in circles. Who's right?

Standard approaches (majority vote, Borda count, averaging) give you *an answer* but not the *structure* of the disagreement. You can't tell if the circularity is fundamental (genuine value conflict) or resolvable (one agent is just slightly misaligned).

## The Insight

Hodge decomposition splits any edge flow on a graph into three orthogonal components:

1. **Gradient flow** — differences that arise from a global ranking function. If edge (i→j) has gradient flow, it means j is genuinely "better" than i according to some underlying score. **Resolvable** by finding the scores.

2. **Curl flow** — cyclic disagreements that go around a loop. A says B > C, B says C > A, C says A > B. These are locally inconsistent but *may* resolve when you consider the global structure. **Partially resolvable**.

3. **Harmonic flow** — the component that's in the kernel of both the gradient operator and its adjoint. These disagreements are **irreconcilable** — they reflect fundamentally different value systems that no amount of negotiation will resolve.

The decomposition is unique and orthogonal: ||flow||² = ||gradient||² + ||curl||² + ||harmonic||². The ratio tells you what fraction of disagreement is fixable.

## How It Works

### Build a disagreement graph

```rust
use hodge_consensus::{WeightedGraph, EdgeFlow};

let graph = WeightedGraph::from_edges(4, &[
    (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0),
]);

// Edge flow: how much agent i disagrees with agent j
// Positive = i rates higher than j, negative = opposite
let flow = EdgeFlow::new(graph, vec![0.5, -0.3, 0.8, 0.2]);
```

### Decompose

```rust
use hodge_consensus::HodgeDecomposition;

let decomp = HodgeDecomposition::compute(&flow);
println!("Gradient energy: {:.3} (resolvable)", decomp.gradient_energy());
println!("Curl energy:      {:.3} (cyclic)", decomp.curl_energy());
println!("Harmonic energy:  {:.3} (irreconcilable)", decomp.harmonic_energy());
println!("Residual:         {:.2e}", decomp.residual);
```

### Predict which disputes resolve

```rust
use hodge_consensus::ConsensusPredictor;

let predictor = ConsensusPredictor::from_decomposition(&decomp);
println!("Resolvable edges:   {:?}", predictor.resolvable_edges());
println!("Irreconcilable:     {:?}", predictor.irreconcilable_edges());
```

Edges dominated by gradient flow will resolve once you find the right ranking. Edges dominated by harmonic flow never will — the agents genuinely disagree at a value level.

### Aggregate rankings

```rust
use hodge_consensus::RankAggregation;

let ranking = RankAggregation::from_flow(&flow);
println!("Optimal scores: {:?}", ranking.scores);
println!("Kendall τ:      {:.3} (1 = perfect agreement)", ranking.kendall_tau);
```

The optimal ranking is the least-squares solution to the gradient component — the scores that best explain the resolvable disagreements.

## The Math

For a graph with incidence matrix B, any edge flow f decomposes as:

```
f = B·s + (B^T)·ω + h
```

where s is a node scalar field (scores), ω is an edge 2-form (curl), and h is harmonic (in ker(B) ∩ ker(B^T)). The projection is computed via least-squares.

## Module Map

| Module | What it does |
|---|---|
| `graph` | `WeightedGraph` — adjacency, incidence matrix, Laplacian, triangle detection |
| `flow` | `EdgeFlow` — disagreement values on edges |
| `decomposition` | `HodgeDecomposition` — orthogonal gradient/curl/harmonic split |
| `consensus` | `ConsensusPredictor` — classify edges as resolvable vs irreconcilable |
| `ranking` | `RankAggregation` — optimal ranking from pairwise comparisons |
| `error` | `HodgeError` |

## When To Use This

- **Multi-agent consensus**: understand *why* agents disagree, not just *that* they disagree
- **Rank aggregation**: combine multiple rankings (agents, judges, models) into one optimal ranking
- **Conflict resolution**: distinguish fixable miscommunications from fundamental value conflicts
- **Recommendation systems**: decompose user preference disagreements into systematic (gradient) vs random (harmonic)

## Links

- [Documentation](https://docs.rs/hodge-consensus)
- [Repository](https://github.com/SuperInstance/hodge-consensus)
- [crates.io](https://crates.io/crates/hodge-consensus)
- Jiang et al. (2011) — *Statistical ranking and combinatorial Hodge theory*

## License

MIT
