use crate::error::HodgeError;
use crate::graph::WeightedGraph;
use serde::{Deserialize, Serialize};

/// A flow (disagreement values) on the edges of a graph.
///
/// `values[k]` is the signed flow on `graph.edges[k]`. Positive means
/// the "opinion" at node j is higher than node i (for edge (i,j)).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeFlow {
    pub graph: WeightedGraph,
    pub values: Vec<f64>,
}

impl EdgeFlow {
    /// Create a new edge flow, validating length matches edge count.
    pub fn new(graph: WeightedGraph, values: Vec<f64>) -> Result<Self, HodgeError> {
        if values.len() != graph.edges.len() {
            return Err(HodgeError::FlowLengthMismatch {
                flow_len: values.len(),
                edge_count: graph.edges.len(),
            });
        }
        Ok(Self { graph, values })
    }

    /// Create from pairwise comparisons: for each (i, j, score_ij), score_ij = opinion_j - opinion_i.
    pub fn from_comparisons(
        graph: WeightedGraph,
        comparisons: &[(usize, usize, f64)],
    ) -> Result<Self, HodgeError> {
        let mut values = vec![0.0; graph.edges.len()];
        for &(i, j, s) in comparisons {
            let (a, b) = if i <= j { (i, j) } else { (j, i) };
            let sign = if i <= j { 1.0 } else { -1.0 };
            let idx = graph
                .edges
                .iter()
                .position(|&(u, v, _)| u == a && v == b)
                .ok_or(HodgeError::EdgeOutOfBounds {
                    index: usize::MAX,
                    num_edges: graph.edges.len(),
                })?;
            values[idx] += sign * s;
        }
        Self::new(graph, values)
    }

    /// L2 norm of the flow.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}
