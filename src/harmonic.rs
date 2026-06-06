//! Harmonic component analysis.
//!
//! Harmonic disagreements persist because of the graph's topology — specifically,
//! its connected components. Two disconnected groups of agents cannot reach
//! consensus through the graph structure alone.
//!
//! The first Betti number b₁ (dimension of the harmonic space H¹) counts the
//! number of independent persistent disagreements.

use serde::{Deserialize, Serialize};

use crate::decomposition::{EnergyFractions, HodgeComponents};
use crate::graph::OpinionGraph;

/// Analysis of the harmonic (persistent) component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicAnalysis {
    /// The harmonic flow component (one entry per edge).
    pub harmonic_flow: Vec<f64>,
    /// Energy fraction in the harmonic component.
    pub energy_fraction: f64,
    /// Dimension of the harmonic space (≈ number of connected components − 1).
    pub h1_dimension: usize,
    /// Labels of each connected component.
    pub components: Vec<Vec<String>>,
    /// Number of connected components.
    pub n_components: usize,
}

impl HarmonicAnalysis {
    /// Compute harmonic analysis from a decomposition.
    pub fn from_decomposition(
        graph: &OpinionGraph,
        decomp: &HodgeComponents,
    ) -> Self {
        let components = connected_components(graph);
        let n_components = components.len();
        // H¹ dimension = max(0, n_components − 1) for the 0-th cohomology
        // For edge flows (1-cochains), H¹ ≈ independent cycles = edges − nodes + components
        let h1 = if graph.n() > 0 && graph.m() > graph.n() {
            graph.m() - graph.n() + n_components
        } else {
            0
        };

        let fractions = decomp.energy_fractions();

        HarmonicAnalysis {
            harmonic_flow: decomp.harmonic.clone(),
            energy_fraction: fractions.harmonic,
            h1_dimension: h1,
            components,
            n_components,
        }
    }

    /// Whether the graph topology permits full consensus (single component, low harmonic).
    pub fn can_reach_consensus(&self, threshold: f64) -> bool {
        self.n_components == 1 && self.energy_fraction < threshold
    }

    /// Identify which agents are in disconnected components (potential splinter groups).
    pub fn isolated_agents(&self) -> Vec<String> {
        self.components
            .iter()
            .filter(|c| c.len() == 1)
            .flat_map(|c| c.clone())
            .collect()
    }
}

/// Find connected components using union-find.
fn connected_components(graph: &OpinionGraph) -> Vec<Vec<String>> {
    let n = graph.n();
    if n == 0 {
        return Vec::new();
    }

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    let idx = graph.index_map();
    for (src, dst, _) in &graph.edges {
        let i = idx[src];
        let j = idx[dst];
        let ri = find(&mut parent, i);
        let rj = find(&mut parent, j);
        if ri != rj {
            parent[ri] = rj;
        }
    }

    let mut groups: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for (i, agent) in graph.agents.iter().enumerate() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(agent.clone());
    }

    let mut result: Vec<Vec<String>> = groups.into_values().collect();
    // Sort by size descending
    result.sort_by(|a, b| b.len().cmp(&a.len()));
    result
}
