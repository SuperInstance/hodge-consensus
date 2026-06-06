//! Consensus protocol based on Hodge decomposition.
//!
//! The consensus protocol projects each agent's opinion onto the gradient
//! (globally consistent) component. Agents update toward gradient flow,
//! achieving consensus when the curl component is small.

use serde::{Deserialize, Serialize};

use crate::decomposition::HodgeComponents;
use crate::graph::OpinionGraph;

/// Result of running a consensus protocol step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusState {
    /// Current node potentials (opinion scores).
    pub potentials: Vec<f64>,
    /// Convergence metric: L² norm of the update step.
    pub convergence_metric: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether consensus was reached.
    pub reached: bool,
    /// Final mean opinion (consensus value).
    pub consensus_value: f64,
}

/// Configuration for the consensus protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Learning rate for gradient updates.
    pub learning_rate: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (L² norm of update).
    pub tolerance: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }
}

/// Run consensus to find the globally consistent ranking of agents.
///
/// The protocol iteratively updates node potentials by following the
/// gradient flow derived from the Hodge decomposition.
pub fn run_consensus(
    graph: &OpinionGraph,
    decomp: &HodgeComponents,
) -> ConsensusState {
    run_consensus_with_config(graph, decomp, ConsensusConfig::default())
}

/// Run consensus with custom configuration.
pub fn run_consensus_with_config(
    graph: &OpinionGraph,
    decomp: &HodgeComponents,
    config: ConsensusConfig,
) -> ConsensusState {
    let n = graph.n();
    if n == 0 {
        return ConsensusState {
            potentials: Vec::new(),
            convergence_metric: 0.0,
            iterations: 0,
            reached: true,
            consensus_value: 0.0,
        };
    }

    let idx = graph.index_map();
    let lap = graph.laplacian();

    // Initialize potentials from the gradient component.
    let mut phi = vec![0.0; n];
    for (src, dst, _) in &graph.edges {
        let i = idx[src];
        let j = idx[dst];
        // Use gradient flow to initialize
    }

    // Actually use the decomposition gradient to seed potentials
    // Solve L φ = B^T f for the gradient potential
    let flow = graph.flow();
    let mut btf = vec![0.0; n];
    for (k, (src, dst, _)) in graph.edges.iter().enumerate() {
        let i = idx[src];
        let j = idx[dst];
        let s = graph.edges[k].2.abs().sqrt().max(0.0);
        btf[i] += s * flow[k];
        btf[j] -= s * flow[k];
    }

    // Iterative solver: Jacobi-like updates
    let mut iter = 0;
    let mut converged = false;

    for _ in 0..config.max_iterations {
        iter += 1;
        let mut new_phi = phi.clone();
        let mut max_delta = 0.0f64;

        for i in 0..n {
            if lap[i][i] < 1e-15 {
                continue;
            }
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += lap[i][j] * phi[j];
                }
            }
            let update = config.learning_rate * ((btf[i] - sum) / lap[i][i] - phi[i]);
            new_phi[i] += update;
            max_delta = max_delta.max(update.abs());
        }

        phi = new_phi;
        if max_delta < config.tolerance {
            converged = true;
            break;
        }
    }

    // Compute convergence metric
    let metric = phi
        .iter()
        .zip(lap.iter().map(|row| {
            phi.iter()
                .zip(row.iter())
                .map(|(p, l)| p * l)
                .sum::<f64>()
        }))
        .map(|(p, lp)| (lp - btf[phi.iter().position(|&x| (x - *p).abs() < 1e-10).unwrap_or(0)]) .powi(2))
        .sum::<f64>()
        .sqrt();

    let consensus_value = if n > 0 {
        phi.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    ConsensusState {
        potentials: phi,
        convergence_metric: metric,
        iterations: iter,
        reached: converged,
        consensus_value,
    }
}

/// Compute the DeGroot-style consensus value (weighted average).
pub fn degroot_consensus(graph: &OpinionGraph, initial_opinions: &[f64]) -> f64 {
    let n = graph.n();
    if n == 0 {
        return 0.0;
    }

    let idx = graph.index_map();

    // Build stochastic matrix from graph
    let mut p = vec![vec![0.0; n]; n];
    for (src, dst, w) in &graph.edges {
        let i = idx[src];
        let j = idx[dst];
        p[i][j] += w.abs();
    }
    // Normalize rows
    for i in 0..n {
        let row_sum: f64 = p[i].iter().sum();
        if row_sum > 1e-15 {
            for j in 0..n {
                p[i][j] /= row_sum;
            }
        } else {
            p[i][i] = 1.0; // isolated node
        }
    }

    // Power iteration to find stationary distribution
    let mut pi = vec![1.0 / n as f64; n];
    for _ in 0..500 {
        let mut new_pi = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_pi[i] += pi[j] * p[j][i];
            }
        }
        let norm: f64 = new_pi.iter().sum();
        if norm > 1e-15 {
            for x in &mut new_pi {
                *x /= norm;
            }
        }
        pi = new_pi;
    }

    // Consensus = π · initial_opinions
    initial_opinions
        .iter()
        .zip(&pi)
        .map(|(o, w)| o * w)
        .sum()
}
