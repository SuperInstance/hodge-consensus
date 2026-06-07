# hodge-consensus

Hodge decomposition of multi-agent disagreements. Every opinion flow on a graph splits into three orthogonal components — gradient (resolvable), curl (cyclic), and harmonic (irreconcilable). The Hodge theorem tells you which arguments can be won.

```
opinion flow = gradient + curl + harmonic
```

- **Gradient**: globally consistent pairwise differences. Someone is right, and the Hodge potential says who.
- **Curl**: cyclic disagreements (A > B > C > A). They loop, but with enough discussion they converge.
- **Harmonic**: topological obstructions. These agents will never agree — the graph structure prevents it.

## Install

```toml
[dependencies]
hodge-consensus = "0.1.0"
```

## The 30-Line Demo

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn main() {
    let mut g = OpinionGraph::new();
    g.add_symmetric_edge("alice", "bob", 0.8);
    g.add_symmetric_edge("bob", "carol", 0.6);
    g.add_symmetric_edge("carol", "alice", 0.3);

    let decomp = HodgeComponents::decompose(&g);
    let norms = decomp.norms();
    println!("Gradient: {:.3}  (resolvable)", norms.gradient_norm);
    println!("Curl:     {:.3}  (cyclic)", norms.curl_norm);
    println!("Harmonic: {:.3}  (irreconcilable)", norms.harmonic_norm);

    let frac = decomp.energy_fractions();
    println!("{:.0}% of disagreement energy is in the gradient (solvable)", frac.gradient * 100.0);
}
```

## Architecture

```
hodge-consensus
├── lib.rs           — Re-exports + integration tests
├── graph.rs         — OpinionGraph: directed weighted agent graph
├── decomposition.rs — HodgeComponents: gradient + curl + harmonic
├── harmonic.rs      — HarmonicAnalysis: connected components, H¹ dimension
├── consensus.rs     — ConsensusState: iterative convergence protocol
├── prediction.rs    — DisputePrediction: will this argument resolve?
└── ranking.rs       — AgentRanking: who is most agreeable?
```

## Opinion Graphs

The `OpinionGraph` models agents as nodes and agreement strengths as directed edges:

```rust
use hodge_consensus::OpinionGraph;

fn build_graph() {
    let mut g = OpinionGraph::new();

    // Add agents explicitly
    g.add_agent("alice");
    g.add_agent("bob");

    // Add directed edges (alice agrees with bob at strength 0.9)
    g.add_edge("alice", "bob", 0.9);

    // Symmetric edges (both directions same weight)
    g.add_symmetric_edge("bob", "carol", 0.7);

    println!("Agents: {} (alice, bob, carol)", g.n());
    println!("Edges:  {}", g.m()); // 3 (alice→bob, bob→carol, carol→bob)
}
```

### Graph Constructors

```rust
use hodge_consensus::OpinionGraph;

fn graph_builders() {
    // Complete graph: every agent agrees with every other at weight 1.0
    let complete = OpinionGraph::complete(5, 1.0);
    println!("Complete(5): {} agents, {} edges", complete.n(), complete.m()); // 5, 20

    // Ring graph: each agent agrees with the next
    let ring = OpinionGraph::ring(5, 1.0);
    println!("Ring(5): {} agents, {} edges", ring.n(), ring.m()); // 5, 5
}
```

### Graph Matrices

```rust
use hodge_consensus::OpinionGraph;

fn graph_matrices() {
    let mut g = OpinionGraph::new();
    g.add_edge("a", "b", 2.0);
    g.add_edge("a", "c", 3.0);
    g.add_edge("b", "c", 1.0);

    // Laplacian L = D - A (n×n)
    let lap = g.laplacian();
    println!("Laplacian: {:?}", lap);

    // Adjacency matrix (n×n)
    let adj = g.adjacency();
    println!("Adjacency: {:?}", adj);

    // Degree matrix (diagonal n×n)
    let deg = g.degree_matrix();
    println!("Degree: {:?}", deg);

    // Edge-incidence matrix B (m×n)
    let inc = g.incidence();
    println!("Incidence: {:?}", inc);

    // Opinion flow vector (length m)
    let flow = g.flow();
    println!("Flow: {:?}", flow); // [2.0, 3.0, 1.0]
}
```

## Hodge Decomposition

The core operation. Given an opinion graph, decompose every edge flow:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn full_decomposition() {
    let mut g = OpinionGraph::new();
    g.add_symmetric_edge("alice", "bob", 0.9);
    g.add_symmetric_edge("bob", "carol", 0.7);
    g.add_symmetric_edge("carol", "dave", 0.4);
    g.add_symmetric_edge("dave", "alice", 0.2);

    let decomp = HodgeComponents::decompose(&g);

    // Each edge gets three component values
    for k in 0..g.m() {
        println!("Edge {}: gradient={:.3} curl={:.3} harmonic={:.3}",
            k, decomp.gradient[k], decomp.curl[k], decomp.harmonic[k]);
    }

    // Reconstruction: gradient + curl + harmonic = total
    let reconstructed = decomp.reconstruct();
    for k in 0..g.m() {
        assert!((reconstructed[k] - decomp.total[k]).abs() < 1e-6);
    }
}
```

### Energy Analysis

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn energy_analysis() {
    let g = OpinionGraph::complete(4, 0.7);
    let decomp = HodgeComponents::decompose(&g);

    // Norms of each component
    let norms = decomp.norms();
    println!("‖gradient‖ = {:.4}", norms.gradient_norm);
    println!("‖curl‖     = {:.4}", norms.curl_norm);
    println!("‖harmonic‖ = {:.4}", norms.harmonic_norm);
    println!("‖total‖    = {:.4}", norms.total_norm);

    // Energy fractions (should sum to ~1.0)
    let frac = decomp.energy_fractions();
    println!("Gradient energy: {:.1}%", frac.gradient * 100.0);
    println!("Curl energy:     {:.1}%", frac.curl * 100.0);
    println!("Harmonic energy: {:.1}%", frac.harmonic * 100.0);
    println!("Sum:             {:.3}", frac.gradient + frac.curl + frac.harmonic);

    // Orthogonality verification
    let report = decomp.verify_orthogonality();
    println!("⟨g,c⟩={:.6} ⟨g,h⟩={:.6} ⟨c,h⟩={:.6} orthogonal={}",
        report.gradient_dot_curl,
        report.gradient_dot_harmonic,
        report.curl_dot_harmonic,
        report.is_orthogonal);
}
```

## Predicting Which Disputes Resolve

This is the killer feature. For each disagreement (edge), predict whether it will resolve:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};
use hodge_consensus::prediction::{predict_all, predict_edge, will_resolve};

fn predict_disputes() {
    let mut g = OpinionGraph::new();
    g.add_symmetric_edge("alice", "bob", 0.9);   // strong agreement
    g.add_symmetric_edge("bob", "carol", -0.3);  // disagreement
    g.add_symmetric_edge("carol", "alice", 0.5); // moderate agreement

    let decomp = HodgeComponents::decompose(&g);

    // Predict all edges at once
    let report = predict_all(&decomp);
    println!("Resolvability: {:.0}%", report.resolvability * 100.0);
    println!("Resolvable: {}, Persistent: {}",
        report.n_resolvable, report.n_persistent);

    for pred in &report.predictions {
        println!("Edge {}: {} (dominant: {}, confidence: {:.0}%)",
            pred.edge_index,
            if pred.will_resolve { "WILL RESOLVE" } else { "PERSISTS" },
            pred.dominant_component,
            pred.confidence * 100.0);
    }

    // Quick check for a single edge
    println!("Edge 0 will resolve: {}", will_resolve(&decomp, 0));
}
```

### Resolution Logic

The prediction follows the Hodge structure:

| Dominant component | Will resolve? | Why |
|---|---|---|
| Gradient | Yes | Globally consistent — there's a correct answer |
| Curl | Yes (moderate confidence) | Cyclic but converges under discussion |
| Harmonic | No | Topological obstruction — can't be resolved |

## Agent Ranking: Who Is Most Agreeable?

Rank agents by how well their opinions align with the gradient (globally consistent) component:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};
use hodge_consensus::ranking::rank_agents;

fn rank_the_room() {
    let mut g = OpinionGraph::new();
    g.add_symmetric_edge("alice", "bob", 0.9);
    g.add_symmetric_edge("bob", "carol", 0.5);
    g.add_symmetric_edge("carol", "alice", 0.1);
    g.add_symmetric_edge("alice", "dave", 0.8);
    g.add_symmetric_edge("bob", "dave", 0.7);
    g.add_symmetric_edge("carol", "dave", -0.3);

    let decomp = HodgeComponents::decompose(&g);
    let report = rank_agents(&g, &decomp);

    println!("Agent Rankings (by agreeability):");
    for r in &report.rankings {
        println!("  #{}: {} — agreeability={:.2} gradient_alignment={:.2}",
            r.rank, r.agent, r.agreeability, r.gradient_alignment);
    }

    println!("\nCooperators (top quartile): {:?}", report.cooperators);
    println!("Contrarians (bottom quartile): {:?}", report.contrarians);
}
```

### Single Agent Query

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};
use hodge_consensus::ranking::agent_agreeability;

fn check_agent() {
    let g = OpinionGraph::complete(3, 1.0);
    let decomp = HodgeComponents::decompose(&g);

    let score = agent_agreeability(&g, &decomp, "agent_0");
    println!("Agent 0 agreeability: {:.2}", score); // 0.0 to 1.0

    // Unknown agent returns 0.0
    let unknown = agent_agreeability(&g, &decomp, "ghost");
    assert_eq!(unknown, 0.0);
}
```

## Harmonic Analysis: Connected Components

The harmonic component reveals the graph's topology:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, HarmonicAnalysis};

fn topology_check() {
    // Disconnected: two separate pairs
    let mut g = OpinionGraph::new();
    g.add_edge("a", "b", 1.0);
    g.add_edge("c", "d", 1.0);

    let decomp = HodgeComponents::decompose(&g);
    let ha = HarmonicAnalysis::from_decomposition(&g, &decomp);

    println!("Connected components: {}", ha.n_components); // 2
    println!("Components: {:?}", ha.components);           // [["a","b"], ["c","d"]]
    println!("H¹ dimension: {}", ha.h1_dimension);        // 0

    // Can these agents reach consensus? No — they're disconnected
    assert!(!ha.can_reach_consensus(0.1));

    // Isolated agents (components of size 1)
    let mut g2 = OpinionGraph::new();
    g2.add_agent("loner");
    g2.add_edge("a", "b", 1.0);
    let decomp2 = HodgeComponents::decompose(&g2);
    let ha2 = HarmonicAnalysis::from_decomposition(&g2, &decomp2);
    assert!(ha2.isolated_agents().contains(&"loner".to_string()));
}
```

### H¹ Dimension (Independent Cycles)

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, HarmonicAnalysis};

fn cycle_space() {
    // Ring with extra chord: creates independent cycles
    let mut g = OpinionGraph::ring(5, 1.0);
    g.add_edge("agent_0", "agent_2", 0.5);
    // 6 edges - 5 nodes + 1 component = 2

    let decomp = HodgeComponents::decompose(&g);
    let ha = HarmonicAnalysis::from_decomposition(&g, &decomp);
    println!("H¹ dimension: {}", ha.h1_dimension); // 2
}
```

## Consensus Protocol

The consensus protocol iteratively updates agent opinions toward the gradient component:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, ConsensusState, ConsensusConfig};
use hodge_consensus::consensus::{run_consensus, run_consensus_with_config, degroot_consensus};

fn consensus_demo() {
    let g = OpinionGraph::complete(4, 1.0);
    let decomp = HodgeComponents::decompose(&g);

    // Default consensus
    let state = run_consensus(&g, &decomp);
    println!("Converged: {} ({} iterations)", state.reached, state.iterations);
    println!("Consensus value: {:.4}", state.consensus_value);
    println!("Potentials: {:?}", state.potentials);

    // Custom configuration
    let config = ConsensusConfig {
        learning_rate: 0.5,
        max_iterations: 100,
        tolerance: 1e-6,
    };
    let state = run_consensus_with_config(&g, &decomp, config);
    println!("Custom config converged in {} iterations", state.iterations);
}
```

### DeGroot Consensus (Weighted Average)

```rust
use hodge_consensus::OpinionGraph;
use hodge_consensus::consensus::degroot_consensus;

fn degroot() {
    let g = OpinionGraph::complete(3, 1.0);
    let opinions = vec![1.0, 2.0, 3.0];

    let consensus = degroot_consensus(&g, &opinions);
    println!("DeGroot consensus: {:.2}", consensus);
    // Close to the mean (2.0) for a complete uniform graph
    assert!((consensus - 2.0).abs() < 0.5);
}
```

## Full Pipeline Example

```rust
use hodge_consensus::{
    OpinionGraph, HodgeComponents, HarmonicAnalysis,
    ConsensusState, ConsensusConfig,
};
use hodge_consensus::ranking::rank_agents;
use hodge_consensus::prediction::predict_all;
use hodge_consensus::consensus::run_consensus;

fn full_pipeline() {
    // 1. Build the opinion graph
    let mut g = OpinionGraph::new();
    g.add_symmetric_edge("alice", "bob", 0.9);
    g.add_symmetric_edge("bob", "carol", 0.7);
    g.add_symmetric_edge("carol", "dave", 0.4);
    g.add_symmetric_edge("dave", "alice", 0.2);

    // 2. Decompose
    let decomp = HodgeComponents::decompose(&g);
    let norms = decomp.norms();
    println!("=== Decomposition ===");
    println!("Gradient: {:.3}  Curl: {:.3}  Harmonic: {:.3}",
        norms.gradient_norm, norms.curl_norm, norms.harmonic_norm);

    // 3. Topology check
    let ha = HarmonicAnalysis::from_decomposition(&g, &decomp);
    println!("\n=== Topology ===");
    println!("Components: {}", ha.n_components);
    println!("Can reach consensus: {}", ha.can_reach_consensus(0.5));

    // 4. Agent ranking
    let rankings = rank_agents(&g, &decomp);
    println!("\n=== Rankings ===");
    for r in &rankings.rankings {
        println!("  #{}: {} (agreeability: {:.2})", r.rank, r.agent, r.agreeability);
    }

    // 5. Dispute prediction
    let predictions = predict_all(&decomp);
    println!("\n=== Predictions ===");
    println!("Resolvability: {:.0}%", predictions.resolvability * 100.0);
    for p in &predictions.predictions {
        println!("  Edge {}: {} ({})", p.edge_index,
            if p.will_resolve { "RESOLVE" } else { "PERSIST" },
            p.dominant_component);
    }

    // 6. Consensus
    let state = run_consensus(&g, &decomp);
    println!("\n=== Consensus ===");
    println!("Converged: {} in {} iterations", state.reached, state.iterations);
    println!("Consensus value: {:.4}", state.consensus_value);
}
```

## Serialization

All major types support serde:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn serde_example() {
    let mut g = OpinionGraph::new();
    g.add_edge("a", "b", 1.0);
    g.add_edge("b", "c", 0.5);

    let decomp = HodgeComponents::decompose(&g);

    // Serialize to JSON
    let json = serde_json::to_string(&decomp).unwrap();
    println!("{}", json);

    // Deserialize back
    let decomp2: HodgeComponents = serde_json::from_str(&json).unwrap();
    assert_eq!(decomp.total.len(), decomp2.total.len());
}
```

## API Reference

### `graph` Module — `OpinionGraph`

| Method | Description |
|---|---|
| `new()` | Empty graph |
| `add_agent(name)` | Add a node |
| `add_edge(src, dst, weight)` | Directed edge |
| `add_symmetric_edge(a, b, weight)` | Bidirectional edge |
| `n()` / `m()` | Node / edge count |
| `laplacian()` | L = D − A (n×n) |
| `adjacency()` | A (n×n) |
| `degree_matrix()` | D (diagonal n×n) |
| `incidence()` | B (m×n) |
| `flow()` | Edge flow vector |
| `complete(n, w)` | Complete graph |
| `ring(n, w)` | Ring graph |

### `decomposition` Module — `HodgeComponents`

| Method | Description |
|---|---|
| `decompose(graph)` | Compute decomposition |
| `norms()` | ‖g‖, ‖c‖, ‖h‖, ‖total‖ |
| `energy_fractions()` | Fraction in each component |
| `verify_orthogonality()` | Check ⟨g,c⟩ ≈ ⟨g,h⟩ ≈ ⟨c,h⟩ ≈ 0 |
| `reconstruct()` | g + c + h (should equal total) |

### `harmonic` Module — `HarmonicAnalysis`

| Method | Description |
|---|---|
| `from_decomposition(graph, decomp)` | Compute harmonic analysis |
| `can_reach_consensus(threshold)` | Single component + low harmonic? |
| `isolated_agents()` | Agents in singleton components |
| `h1_dimension` | Dimension of harmonic space |

### `prediction` Module

| Function | Description |
|---|---|
| `predict_all(decomp)` | Predict all edges |
| `predict_edge(decomp, idx)` | Predict single edge |
| `will_resolve(decomp, idx)` | Quick boolean check |
| `resolvability_score(decomp)` | Overall resolvability 0..1 |

### `ranking` Module

| Function | Description |
|---|---|
| `rank_agents(graph, decomp)` | Full ranking report |
| `agent_agreeability(graph, decomp, name)` | Single agent score |

### `consensus` Module

| Function | Description |
|---|---|
| `run_consensus(graph, decomp)` | Default config consensus |
| `run_consensus_with_config(graph, decomp, config)` | Custom config |
| `degroot_consensus(graph, opinions)` | Weighted average |

## Building and Testing

```bash
cargo build
cargo test
```

All tests are self-contained — no external services needed.

## License

MIT OR Apache-2.0
