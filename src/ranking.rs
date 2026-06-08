use crate::decomposition::HodgeDecomposition;
use crate::error::HodgeError;
use crate::flow::EdgeFlow;
use crate::graph::WeightedGraph;
use serde::{Deserialize, Serialize};

/// Optimal ranking from pairwise comparisons via Hodge gradient projection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankAggregation {
    /// Node labels.
    pub nodes: Vec<String>,
    /// Optimal score for each node (higher = better rank).
    pub scores: Vec<f64>,
    /// Kendall tau–like agreement metric (1.0 = perfect).
    pub kendall_tau: f64,
}

impl RankAggregation {
    /// Compute consensus ranking from pairwise comparison data.
    ///
    /// `comparisons` are (i, j, score) meaning node j is preferred over node i by `score`.
    /// `labels` provides human-readable names for each node.
    pub fn rank(
        graph: WeightedGraph,
        comparisons: &[(usize, usize, f64)],
        labels: &[String],
    ) -> Result<Self, HodgeError> {
        if labels.len() != graph.n {
            return Err(HodgeError::LabelCountMismatch {
                label_count: labels.len(),
                node_count: graph.n,
            });
        }

        let flow = EdgeFlow::from_comparisons(graph, comparisons)?;
        let decomp = HodgeDecomposition::decompose(&flow)?;

        // Node potentials from gradient solve give us the ranking scores
        let b = flow.graph.incidence_matrix();
        let n = flow.graph.n;
        let _m = flow.graph.edges.len();

        // Reconstruct potentials from gradient flow: solve BᵀB f = Bᵀ gradient_flow
        let lap = flow.graph.laplacian();
        let bt_grad: Vec<f64> = (0..n)
            .map(|j| {
                b.iter()
                    .zip(decomp.gradient_flow.iter())
                    .map(|(row, &g)| row[j] * g)
                    .sum()
            })
            .collect();

        // Solve for potentials
        let scores = crate::decomposition::solve_linear(&lap, &bt_grad, 1e-10);

        // Normalize scores to [0, 1] range
        let min_s = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_s - min_s;
        let scores: Vec<f64> = if range > 1e-15 {
            scores.iter().map(|s| (s - min_s) / range).collect()
        } else {
            vec![0.5; n]
        };

        // Compute Kendall tau: fraction of concordant pairs
        let mut concordant = 0usize;
        let mut total = 0usize;
        for &(i, j, s) in comparisons {
            let (a, b_idx) = if i <= j { (i, j) } else { (j, i) };
            let diff = scores[b_idx] - scores[a];
            let agree = if i <= j { diff * s >= 0.0 } else { -diff * s >= 0.0 };
            if agree {
                concordant += 1;
            }
            total += 1;
        }

        let kendall_tau = if total > 0 {
            concordant as f64 / total as f64
        } else {
            1.0
        };

        Ok(Self {
            nodes: labels.to_vec(),
            scores,
            kendall_tau,
        })
    }

    /// Return node indices sorted by score (descending = best first).
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.scores.len()).collect();
        idx.sort_by(|&a, &b| {
            self.scores[b]
                .partial_cmp(&self.scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }

    /// Return node labels sorted by score (descending).
    pub fn ranking(&self) -> Vec<&str> {
        self.sorted_indices()
            .iter()
            .map(|&i| self.nodes[i].as_str())
            .collect()
    }
}
