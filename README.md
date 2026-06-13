# hodge-consensus

**Hodge decomposition of multi-agent consensus.**

Every disagreement decomposes into three orthogonal components. The Hodge theorem tells you which arguments can be won.

```
opinion flow = gradient + curl + harmonic
```

| Component | Meaning | Resolves? |
|-----------|---------|-----------|
| **Gradient** | Globally consistent — derivable from a single ranking | Already resolved |
| **Curl** | Cyclic disagreement — agents loop but converge | Yes |
| **Harmonic** | Topological obstruction — disconnected groups | No |

## The Idea

Take any group of agents with pairwise opinions. The Hodge decomposition theorem from differential geometry, applied to the discrete setting of graphs, says that their disagreement structure splits cleanly into three pieces:

1. **Gradient (exact) flows** — disagreements that are globally consistent. If Alice > Bob > Carol in pairwise agreement, that's a gradient flow. There exists a single scalar "agreeability" potential that explains all of it.

2. **Curl (co-exact) flows** — cyclic disagreements. Alice disagrees with Bob, Bob with Carol, Carol with Alice. These form loops. They cancel out over time through iterative consensus protocols.

3. **Harmonic flows** — disagreements that persist because of topology. Two disconnected groups that never interact. These are the irreconcilable differences. The first Betti number b₁ tells you how many independent persistent disagreements exist.

This is the same mathematics that tells you why some fluid flows have irrotational components, why some vector fields are conservative, and why some differential forms are closed but not exact. Applied to consensus.

## Quick Start

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

// Build a graph of agent opinions
let mut g = OpinionGraph::new();
g.add_symmetric_edge("alice", "bob", 0.9);
g.add_symmetric_edge("bob", "carol", 0.7);
g.add_symmetric_edge("carol", "dave", 0.4);
g.add_symmetric_edge("dave", "alice", 0.2);

// Decompose
let decomp = HodgeComponents::decompose(&g);
let norms = decomp.norms();

println!("Gradient energy:  {:.3}", norms.gradient_norm);
println!("Curl energy:       {:.3}", norms.curl_norm);
println!("Harmonic energy:   {:.3}", norms.harmonic_norm);

// Predict which disputes will resolve
use hodge_consensus::prediction;
let report = prediction::predict_all(&decomp);
for p in &report.predictions {
    println!("{:?} → will_resolve={}, dominant={}",
        p.will_resolve, p.dominant_component);
}
```

## Modules

### `graph` — Opinion Graphs

```rust
let mut g = OpinionGraph::new();
g.add_edge("alice", "bob", 0.8);        // alice → bob, weight 0.8
g.add_symmetric_edge("bob", "carol", 0.6); // bidirectional

let lap = g.laplacian();    // L = D - A
let adj = g.adjacency();    // weighted adjacency matrix
let inc = g.incidence();    // edge-node incidence matrix
```

Constructors: `OpinionGraph::complete(n, w)`, `OpinionGraph::ring(n, w)`.

### `decomposition` — Hodge Decomposition

```rust
let decomp = HodgeComponents::decompose(&graph);

// The three components (one entry per edge)
decomp.gradient;   // exact flows
decomp.curl;       // cyclic disagreements
decomp.harmonic;   // topological obstructions

// Energy analysis
let frac = decomp.energy_fractions();
assert!((frac.gradient + frac.curl + frac.harmonic - 1.0).abs() < 0.05);

// Verify orthogonality
let ortho = decomp.verify_orthogonality();
assert!(ortho.is_orthogonal);

// Reconstruct: gradient + curl + harmonic ≡ total
let reconstructed = decomp.reconstruct();
```

### `harmonic` — Topological Analysis

```rust
use hodge_consensus::HarmonicAnalysis;

let ha = HarmonicAnalysis::from_decomposition(&graph, &decomp);
println!("Connected components: {}", ha.n_components);
println!("H¹ dimension:          {}", ha.h1_dimension);
println!("Harmonic energy:       {:.3}", ha.energy_fraction);

// Find isolated agents (splinter groups)
let loners = ha.isolated_agents();
```

### `consensus` — Consensus Protocol

```rust
use hodge_consensus::consensus;

let state = consensus::run_consensus(&graph, &decomp);
println!("Consensus value: {:.3}", state.consensus_value);
println!("Iterations:      {}", state.iterations);
println!("Converged:       {}", state.reached);

// DeGroot weighted average
let opinions = vec![1.0, 2.0, 3.0, 4.0];
let consensus = consensus::degroot_consensus(&graph, &opinions);
```

### `ranking` — Agent Agreeability

```rust
use hodge_consensus::ranking;

let report = ranking::rank_agents(&graph, &decomp);
for r in &report.rankings {
    println!("#{} {} — agreeability: {:.3}",
        r.rank, r.agent, r.agreeability);
}

println!("Cooperators: {:?}", report.cooperators);
println!("Contrarians: {:?}", report.contrarians);
```

### `prediction` — Dispute Resolution

```rust
use hodge_consensus::prediction;

let report = prediction::predict_all(&decomp);
println!("Resolvability: {:.1}%", report.resolvability * 100.0);
println!("Resolvable:    {}", report.n_resolvable);
println!("Persistent:    {}", report.n_persistent);

// Single dispute check
let will = prediction::will_resolve(&decomp, edge_index);
```

## How It Works

### The Laplacian

Given a directed weighted graph with adjacency matrix **A** and degree matrix **D**, the combinatorial Laplacian is **L = D − A**. This is the discrete analogue of the Laplace–Beltrami operator.

### The Decomposition

For an edge flow **f** (vector of pairwise opinions), the Hodge decomposition is:

```
f = grad(φ) + curl(ω) + h
```

where:
- `φ` is a scalar potential on nodes (solved via conjugate gradient on Lφ = Bᵀf)
- `ω` is a flow on 2-cells (solved by projecting onto cycle space)
- `h` is harmonic: Lh = 0 and curl(h) = 0

The three components are mutually orthogonal under the natural inner product.

### The Prediction

- **Gradient-dominant disputes** are already resolved — there's a consistent global ranking.
- **Curl-dominant disputes** will resolve — the cyclic structure averages out through iterative updates.
- **Harmonic-dominant disputes** will persist — the graph topology prevents resolution.

## Installation

```toml
[dependencies]
hodge-consensus = "0.1"
```

## API Surface

| Type | Module | Description |
|------|--------|-------------|
| `OpinionGraph` | `graph` | Directed weighted graph of agent opinions |
| `HodgeComponents` | `decomposition` | Gradient + curl + harmonic decomposition |
| `ComponentNorms` | `decomposition` | L² norms of each component |
| `EnergyFractions` | `decomposition` | Proportion of energy in each component |
| `OrthogonalityReport` | `decomposition` | Verification that components are orthogonal |
| `HarmonicAnalysis` | `harmonic` | Topological analysis of persistent disagreements |
| `ConsensusState` | `consensus` | Result of running consensus protocol |
| `ConsensusConfig` | `consensus` | Configuration for convergence parameters |
| `AgentRanking` | `ranking` | Single agent's agreeability score |
| `RankingReport` | `ranking` | Full ranking with cooperators/contrarians |
| `DisputePrediction` | `prediction` | Per-edge resolution prediction |
| `PredictionReport` | `prediction` | Full prediction report with resolvability |

All public types implement `Serialize` and `Deserialize` via serde.

## Why "Hodge"?

William Vallance Douglas Hodge (1903–1975) was a British mathematician who discovered the Hodge decomposition theorem in algebraic geometry. His theorem states that on a compact Riemannian manifold, every differential form decomposes into exact, co-exact, and harmonic parts.

The discrete version on graphs — which this library implements — is sometimes called the **combinatorial Hodge theorem** or **Hodge theory for graphs**. It connects:
- Algebraic topology (cohomology groups)
- Spectral graph theory (Laplacian eigenvalues)
- Optimization (least-squares ranking)
- Social choice (consensus formation)

## License

MIT OR Apache-2.0
