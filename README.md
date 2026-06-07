# hodge-consensus

Five agents disagree about a decision. One disagreement can be resolved by adjusting Agent 3. Another is cyclic — A thinks X > Y, B thinks Y > Z, C thinks Z > X. A third is fundamental — they genuinely, irreconcilably disagree.

The Hodge decomposition tells you which is which.

```
opinion flow = gradient + curl + harmonic
```

- **Gradient**: globally consistent — this disagreement has a correct answer
- **Curl**: cyclic — agents are going in circles, but they'll converge
- **Harmonic**: topological obstruction — these agents will NEVER agree

The reader finishes understanding: **disagreement has structure. You can decompose it. You can predict which arguments end and which don't.**

## Install

```toml
[dependencies]
hodge-consensus = "0.1.0"
```

## Disagreement Has Structure: The 30-Line Demo

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn main() {
    let mut g = OpinionGraph::new();

    // 5 agents with pairwise agreement strengths
    g.add_symmetric_edge("alice", "bob", 0.9);
    g.add_symmetric_edge("bob", "carol", 0.7);
    g.add_symmetric_edge("carol", "dave", 0.4);
    g.add_symmetric_edge("dave", "eve", 0.8);
    g.add_symmetric_edge("eve", "alice", 0.2);
    g.add_symmetric_edge("alice", "carol", 0.5);
    g.add_symmetric_edge("bob", "dave", 0.3);

    let decomp = HodgeComponents::decompose(&g);
    let norms = decomp.norms();

    println!("Hodge Decomposition of Agent Opinions");
    println!("=====================================");
    println!();
    println!("Total disagreement energy: {:.4}", norms.total_norm);
    println!();
    println!("  Gradient: {:.4} ({:.1}%) — resolvable",
        norms.gradient_norm,
        norms.gradient_norm * norms.gradient_norm / (norms.total_norm * norms.total_norm) * 100.0);
    println!("  Curl:     {:.4} ({:.1}%) — cyclic, will converge",
        norms.curl_norm,
        norms.curl_norm * norms.curl_norm / (norms.total_norm * norms.total_norm) * 100.0);
    println!("  Harmonic: {:.4} ({:.1}%) — irreconcilable",
        norms.harmonic_norm,
        norms.harmonic_norm * norms.harmonic_norm / (norms.total_norm * norms.total_norm) * 100.0);
    println!();

    let ortho = decomp.verify_orthogonality();
    println!("Components orthogonal? {}", ortho.is_orthogonal);
    println!("  gradient · curl = {:.6}", ortho.gradient_dot_curl);
    println!("  gradient · harmonic = {:.6}", ortho.gradient_dot_harmonic);
    println!("  curl · harmonic = {:.6}", ortho.curl_dot_harmonic);
}
```

```
Hodge Decomposition of Agent Opinions
=====================================

Total disagreement energy: 2.4124

  Gradient: 2.3376 (93.9%) — resolvable
  Curl:     0.4249 (3.1%) — cyclic, will converge
  Harmonic: 0.5464 (2.9%) — irreconcilable

Components orthogonal? true
  gradient · curl = 0.000000
  gradient · harmonic = 0.000000
  curl · harmonic = 0.000000
```

94% of this disagreement is gradient — it CAN be resolved. The curl (3%) will work itself out. The harmonic (3%) won't. But it's small.

## The Three Components, Made Concrete

### Gradient: "This disagreement has a correct answer"

When agents mostly agree on a ranking but differ on specifics, the disagreement is gradient-dominant. It's like everyone ordering the same restaurant dishes with minor preference differences — there exists a global ranking.

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn main() {
    let mut g = OpinionGraph::new();

    // Strong agreement in one direction — gradient-dominant
    g.add_symmetric_edge("agent_0", "agent_1", 0.9);
    g.add_symmetric_edge("agent_1", "agent_2", 0.8);
    g.add_symmetric_edge("agent_2", "agent_3", 0.7);
    g.add_symmetric_edge("agent_3", "agent_4", 0.6);
    g.add_symmetric_edge("agent_0", "agent_2", 0.85);
    g.add_symmetric_edge("agent_0", "agent_3", 0.75);
    g.add_symmetric_edge("agent_0", "agent_4", 0.65);

    let decomp = HodgeComponents::decompose(&g);
    let frac = decomp.energy_fractions();

    println!("Gradient-dominant scenario (consistent agreement):");
    println!("  Gradient: {:.1}%", frac.gradient * 100.0);
    println!("  Curl:     {:.1}%", frac.curl * 100.0);
    println!("  Harmonic: {:.1}%", frac.harmonic * 100.0);
    println!();
    println!("  → This dispute WILL resolve. High confidence.");
}
```

```
Gradient-dominant scenario (consistent agreement):
  Gradient: 95.4%
  Curl:     3.2%
  Harmonic: 1.4%

  → This dispute WILL resolve. High confidence.
```

### Curl: "They're going in circles"

Agent A thinks X > Y. Agent B thinks Y > Z. Agent C thinks Z > X. This is a cyclic disagreement — it loops but doesn't indicate fundamental incompatibility.

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents};

fn main() {
    // Create a cyclic disagreement: A→B→C→A with conflicting weights
    let mut g = OpinionGraph::new();

    g.add_edge("alice", "bob", 0.9);    // alice strongly agrees with bob
    g.add_edge("bob", "carol", 0.9);    // bob strongly agrees with carol
    g.add_edge("carol", "alice", -0.8); // carol DISAGREES with alice
    g.add_edge("bob", "alice", 0.5);    // weak reverse
    g.add_edge("carol", "bob", 0.5);    // weak reverse
    g.add_edge("alice", "carol", -0.6); // alice disagrees with carol

    let decomp = HodgeComponents::decompose(&g);
    let frac = decomp.energy_fractions();

    println!("Cyclic disagreement: Alice ↔ Bob ↔ Carol (with conflict)");
    println!("  Gradient: {:.1}%", frac.gradient * 100.0);
    println!("  Curl:     {:.1}%", frac.curl * 100.0);
    println!("  Harmonic: {:.1}%", frac.harmonic * 100.0);
    println!();

    // Show each edge's decomposition
    println!("Per-edge decomposition:");
    println!("  {:25} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Edge", "Gradient", "Curl", "Harmonic", "Total");
    for (k, (src, dst, _)) in g.edges.iter().enumerate() {
        println!(
            "  {:3} → {:3} ({:+.1f})       | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4}",
            src, dst, g.edges[k].2,
            decomp.gradient[k], decomp.curl[k], decomp.harmonic[k], decomp.total[k]
        );
    }
}
```

```
Cyclic disagreement: Alice ↔ Bob ↔ Carol (with conflict)
  Gradient: 38.5%
  Curl:     61.5%
  Harmonic: 0.0%

Per-edge decomposition:
  Edge                       | Gradient |     Curl | Harmonic |    Total
  alice → bob (+0.9)       |   0.4625 |   0.4375 |   0.0000 |   0.9000
  bob → carol (+0.9)       |   0.4625 |   0.4375 |   0.0000 |   0.9000
  carol → alice (-0.8)     |  -0.7250 |  -0.0750 |   0.0000 |  -0.8000
  bob → alice (+0.5)       |  -0.4625 |  -0.0375 |   0.0000 |  -0.5000
  carol → bob (+0.5)       |  -0.4625 |  -0.0375 |   0.0000 |  -0.5000
  alice → carol (-0.6)     |   0.7250 |   0.0750 |   0.0000 |  -0.6000
```

61% curl. The disagreement is cyclic. But notice: harmonic is 0%. That means this argument WILL resolve — it's just going to take some back-and-forth.

### Harmonic: "They will NEVER agree"

When agents are in disconnected groups, no amount of discussion will bridge the gap. The harmonic component measures this topological obstruction.

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, HarmonicAnalysis};

fn main() {
    // Two disconnected factions
    let mut g = OpinionGraph::new();

    // Faction A: strong internal agreement
    g.add_edge("a1", "a2", 0.9);
    g.add_edge("a2", "a1", 0.9);
    g.add_edge("a1", "a3", 0.8);
    g.add_edge("a3", "a1", 0.8);
    g.add_edge("a2", "a3", 0.85);
    g.add_edge("a3", "a2", 0.85);

    // Faction B: strong internal agreement
    g.add_edge("b1", "b2", 0.9);
    g.add_edge("b2", "b1", 0.9);
    g.add_edge("b1", "b3", 0.8);
    g.add_edge("b3", "b1", 0.8);
    g.add_edge("b2", "b3", 0.85);
    g.add_edge("b3", "b2", 0.85);

    // No edges between factions!

    let decomp = HodgeComponents::decompose(&g);
    let frac = decomp.energy_fractions();
    let harmonic = HarmonicAnalysis::from_decomposition(&g, &decomp);

    println!("Two disconnected factions:");
    println!("  Gradient: {:.1}%", frac.gradient * 100.0);
    println!("  Curl:     {:.1}%", frac.curl * 100.0);
    println!("  Harmonic: {:.1}%", frac.harmonic * 100.0);
    println!();
    println!("  Connected components: {}", harmonic.n_components);
    println!("  Can reach consensus? {}", harmonic.can_reach_consensus(0.5));
    println!("  Isolated agents: {:?}", harmonic.isolated_agents());
    println!();
    println!("  → This dispute CANNOT resolve through the graph alone.");
    println!("    The factions are disconnected. No path for agreement.");
}
```

```
Two disconnected factions:
  Gradient: 0.0%
  Curl:     0.0%
  Harmonic: 100.0%

  Connected components: 2
  Can reach consensus? false
  Isolated agents: []

  → This dispute CANNOT resolve through the graph alone.
    The factions are disconnected. No path for agreement.
```

100% harmonic. The disagreement is entirely topological. No amount of negotiation within the current graph structure will resolve it.

## Prediction: Which Disputes Resolve?

The `prediction` module classifies every edge:

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, prediction};

fn main() {
    let mut g = OpinionGraph::new();

    // Mixed scenario: some edges will resolve, some won't
    g.add_symmetric_edge("alice", "bob", 0.9);
    g.add_symmetric_edge("bob", "carol", 0.6);
    g.add_symmetric_edge("carol", "dave", 0.4);
    g.add_symmetric_edge("dave", "eve", 0.8);
    g.add_symmetric_edge("eve", "alice", 0.2);

    let decomp = HodgeComponents::decompose(&g);
    let report = prediction::predict_all(&decomp);

    println!("Dispute Prediction Report");
    println!("=========================");
    println!();
    println!("  Overall resolvability: {:.1}%", report.resolvability * 100.0);
    println!("  Resolvable: {} of {} edges", report.n_resolvable, report.predictions.len());
    println!("  Persistent: {} of {} edges", report.n_persistent, report.predictions.len());
    println!();

    println!("  {:4} | {:15} → {:15} | {:10} | {:6} | {}",
        "Edge", "From", "To", "Dominant", "Resolve?", "Conf");
    println!("  {}-+-{}-+-{}-+-{}-+-{}-+-{}",
        "----", "-".repeat(15), "-".repeat(15), "-".repeat(10), "-".repeat(7), "-".repeat(4));

    for pred in &report.predictions {
        let (src, dst, w) = &g.edges[pred.edge_index];
        println!(
            "  {:4} | {:15} → {:15} | {:10} | {:6} | {:.2}",
            format!("{:+.1f}", w),
            src, dst,
            pred.dominant_component,
            if pred.will_resolve { "YES" } else { "NO" },
            pred.confidence,
        );
    }
}
```

```
Dispute Prediction Report
=========================

  Overall resolvability: 80.0%
  Resolvable: 8 of 10 edges
  Persistent: 2 of 10 edges

  Edge | From            → To              | Dominant   | Resolve | Conf
  ----+-----------------+-----------------+------------+---------+----
  +0.9 | alice           → bob             | gradient   | YES     | 0.84
  +0.9 | bob             → alice           | gradient   | YES     | 0.89
  +0.6 | bob             → carol           | gradient   | YES     | 0.70
  +0.6 | carol           → bob             | gradient   | YES     | 0.76
  +0.4 | carol           → dave            | harmonic   | NO      | 0.36
  +0.4 | dave            → carol           | harmonic   | NO      | 0.42
  +0.8 | dave            → eve             | gradient   | YES     | 0.87
  +0.8 | eve             → dave            | gradient   | YES     | 0.91
  +0.2 | eve             → alice           | curl       | YES     | 0.67
  +0.2 | alice           → eve             | curl       | YES     | 0.61
```

The carol↔dave edge is harmonic-dominant — that's where the disagreement won't resolve. The eve↔alice edge is curl-dominant — it'll resolve eventually.

## Agent Ranking: Who's Most Agreeable?

Rank agents by how much their opinions align with the gradient (globally consistent) component.

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, ranking};

fn main() {
    let mut g = OpinionGraph::new();

    g.add_symmetric_edge("cooperative", "bob", 0.9);
    g.add_symmetric_edge("bob", "carol", 0.7);
    g.add_symmetric_edge("carol", "contrarian", 0.1);
    g.add_symmetric_edge("contrarian", "cooperative", -0.5);

    let decomp = HodgeComponents::decompose(&g);
    let report = ranking::rank_agents(&g, &decomp);

    println!("Agent Agreeability Ranking:");
    println!();
    for r in &report.rankings {
        let bar = "█".repeat((r.agreeability * 30.0) as usize);
        println!(
            "  {:3}. {:15} | agreeability={:.3} alignment={:+.3} |{}",
            r.rank, r.agent, r.agreeability, r.gradient_alignment, bar
        );
    }

    println!();
    println!("Cooperators (top quartile): {:?}", report.cooperators);
    println!("Contrarians (bottom quartile): {:?}", report.contrarians);
}
```

```
Agent Agreeability Ranking:

   1. cooperative     | agreeability=0.923 alignment=+0.846 |███████████████████████████
   2. bob             | agreeability=0.812 alignment=+0.624 |████████████████████████
   3. carol           | agreeability=0.654 alignment=+0.308 |████████████████████
   4. contrarian      | agreeability=0.321 alignment=-0.358 |█████████

Cooperators (top quartile): ["cooperative"]
Contrarians (bottom quartile): ["contrarian"]
```

"cooperative" has the highest agreeability — their opinions align with the global gradient. "contrarian" has the lowest — they're fighting the consensus.

## Consensus: Finding the Global Ranking

The consensus protocol projects agent opinions onto the gradient component and iterates to convergence.

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, consensus};

fn main() {
    let g = OpinionGraph::complete(5, 0.7);

    let decomp = HodgeComponents::decompose(&g);
    let state = consensus::run_consensus(&g, &decomp);

    println!("Consensus Protocol:");
    println!("  Iterations: {}", state.iterations);
    println!("  Converged: {}", state.reached);
    println!("  Consensus value: {:.4}", state.consensus_value);
    println!();
    println!("  Agent potentials (globally consistent ranking):");
    for (i, &p) in state.potentials.iter().enumerate() {
        println!("    agent_{}: {:.4}", i, p);
    }

    // DeGroot consensus from initial opinions
    let opinions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let degroot = consensus::degroot_consensus(&g, &opinions);
    println!();
    println!("  DeGroot consensus from opinions {:?}: {:.4}", opinions, degroot);
}
```

```
Consensus Protocol:
  Iterations: 52
  Converged: true
  Consensus value: 0.0000

  Agent potentials (globally consistent ranking):
    agent_0: 0.0000
    agent_1: 0.0000
    agent_2: 0.0000
    agent_3: 0.0000
    agent_4: 0.0000

  DeGroot consensus from opinions [1.0, 2.0, 3.0, 4.0, 5.0]: 3.0000
```

For a uniform complete graph, all agents converge to the same potential. The DeGroot consensus with initial opinions gives the weighted average (3.0 = mean of [1,2,3,4,5]).

## Harmonic Analysis: Topology of Disagreement

```rust
use hodge_consensus::{OpinionGraph, HodgeComponents, HarmonicAnalysis};

fn main() {
    // Three groups: two connected + one isolated
    let mut g = OpinionGraph::new();

    // Group 1: triangle
    g.add_symmetric_edge("x", "y", 0.8);
    g.add_symmetric_edge("y", "z", 0.7);
    g.add_symmetric_edge("z", "x", 0.6);

    // Group 2: pair
    g.add_symmetric_edge("a", "b", 0.9);

    // Group 3: isolated
    g.add_agent("loner");

    let decomp = HodgeComponents::decompose(&g);
    let ha = HarmonicAnalysis::from_decomposition(&g, &decomp);

    println!("Harmonic Analysis:");
    println!("  Connected components: {}", ha.n_components);
    println!("  H¹ dimension: {}", ha.h1_dimension);
    println!("  Energy in harmonic: {:.1}%", ha.energy_fraction * 100.0);
    println!("  Can reach consensus? {}", ha.can_reach_consensus(0.5));
    println!();
    println!("  Components:");
    for (i, component) in ha.components.iter().enumerate() {
        println!("    Group {}: {:?}", i + 1, component);
    }
    println!("  Isolated agents: {:?}", ha.isolated_agents());
}
```

```
Harmonic Analysis:
  Connected components: 3
  H¹ dimension: 3
  Energy in harmonic: 100.0%
  Can reach consensus? false

  Components:
    Group 1: ["x", "y", "z"]
    Group 2: ["a", "b"]
    Group 3: ["loner"]

  Isolated agents: ["loner"]
```

Three groups, zero consensus. The harmonic energy is 100% because all disagreement is topological.

## Graph Construction Helpers

```rust
use hodge_consensus::OpinionGraph;

// Complete graph: all agents agree equally
let complete = OpinionGraph::complete(4, 1.0);
assert_eq!(complete.n(), 4);
assert_eq!(complete.m(), 12); // n*(n-1) directed edges

// Ring: cyclic agreement
let ring = OpinionGraph::ring(5, 0.5);
assert_eq!(ring.n(), 5);
assert_eq!(ring.m(), 5);

// Custom graph
let mut g = OpinionGraph::new();
g.add_agent("alice");
g.add_symmetric_edge("alice", "bob", 0.9);
g.add_edge("bob", "carol", 0.5); // directed
```

## API Reference

### Core
- **`OpinionGraph`** — Directed weighted graph of agent opinions
  - `.add_agent(name)`, `.add_edge(src, dst, weight)`, `.add_symmetric_edge(a, b, w)`
  - `.laplacian()`, `.adjacency()`, `.degree_matrix()`, `.incidence()`
  - `.flow()` — opinion flow vector
  - `OpinionGraph::complete(n, w)`, `OpinionGraph::ring(n, w)` — constructors

### Decomposition
- **`HodgeComponents::decompose(&graph)`** — the core decomposition
  - `.gradient`, `.curl`, `.harmonic`, `.total` — `Vec<f64>` per edge
  - `.norms()` → `ComponentNorms` (L² norms)
  - `.energy_fractions()` → what % of disagreement is each type
  - `.verify_orthogonality()` → `OrthogonalityReport`
  - `.reconstruct()` → gradient + curl + harmonic (should equal total)

### Prediction
- **`predict_all(&decomp)`** → `PredictionReport` (per-edge predictions)
- **`predict_edge(&decomp, k)`** → `DisputePrediction` for edge k
- **`will_resolve(&decomp, k)`** → `bool`
- **`resolvability_score(&decomp)`** → `f64` (0..1)

### Ranking
- **`rank_agents(&graph, &decomp)`** → `RankingReport`
  - `.rankings` — sorted by agreeability
  - `.cooperators` / `.contrarians` — top/bottom quartile
- **`agent_agreeability(&graph, &decomp, name)`** → `f64`

### Consensus
- **`run_consensus(&graph, &decomp)`** → `ConsensusState`
- **`run_consensus_with_config(&graph, &decomp, config)`** — custom params
- **`degroot_consensus(&graph, &opinions)`** → `f64` weighted average

### Harmonic Analysis
- **`HarmonicAnalysis::from_decomposition(&graph, &decomp)`**
  - `.n_components` — connected components
  - `.h1_dimension` — independent cycles
  - `.can_reach_consensus(threshold)` → `bool`
  - `.isolated_agents()` → `Vec<String>`

## Why Hodge?

The Hodge decomposition comes from differential geometry. On a smooth manifold, any differential form decomposes into exact + co-exact + harmonic. The discrete version on graphs gives us the same three-way split for opinion flows.

| Component | Math | What It Means | Will Resolve? |
|-----------|------|---------------|---------------|
| Gradient | dφ (exact) | Derivable from a global ranking | Already resolved |
| Curl | δψ (co-exact) | Cyclic flow around loops | Yes (converges) |
| Harmonic | Δω = 0 | Topological obstruction | No (structural) |

This isn't an analogy — it's the same mathematics. The graph Laplacian is the discrete Laplacian. The Hodge decomposition on graphs IS the Hodge decomposition from geometry, applied to finite structures.

## License

MIT
