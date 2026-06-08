use crate::error::HodgeError;
use serde::{Deserialize, Serialize};

/// A weighted undirected graph with `n` nodes and weighted edges.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedGraph {
    /// Number of nodes.
    pub n: usize,
    /// Edge list: (source, target, weight). We store edges in canonical order (i ≤ j).
    pub edges: Vec<(usize, usize, f64)>,
    /// Dense adjacency matrix (n×n), zero means no edge.
    pub adjacency: Vec<Vec<f64>>,
}

impl WeightedGraph {
    /// Build an empty graph with `n` isolated nodes.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            adjacency: vec![vec![0.0; n]; n],
        }
    }

    /// Add an undirected edge (i, j) with weight `w`.
    /// Returns an error if node indices are out of bounds.
    pub fn add_edge(&mut self, i: usize, j: usize, w: f64) -> Result<(), HodgeError> {
        if i >= self.n || j >= self.n {
            return Err(HodgeError::NodeOutOfBounds {
                index: i.max(j),
                n: self.n,
            });
        }
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        // Update or add edge
        if let Some(pos) = self.edges.iter().position(|&(u, v, _)| u == a && v == b) {
            self.edges[pos].2 += w;
        } else {
            self.edges.push((a, b, w));
        }
        self.adjacency[i][j] += w;
        self.adjacency[j][i] += w;
        Ok(())
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Compute the edge-node incidence matrix B (m × n).
    ///
    /// For oriented edge (i, j) with i < j:
    ///   B[row][i] = -sqrt(weight), B[row][j] = +sqrt(weight)
    pub fn incidence_matrix(&self) -> Vec<Vec<f64>> {
        let m = self.edges.len();
        let n = self.n;
        let mut b = vec![vec![0.0; n]; m];
        for (row, &(i, j, w)) in self.edges.iter().enumerate() {
            let s = w.sqrt();
            b[row][i] = -s;
            b[row][j] = s;
        }
        b
    }

    /// Compute the graph Laplacian L = BᵀB (n × n).
    ///
    /// L[i][i] = sum of weights incident to i
    /// L[i][j] = -weight(i,j) if edge exists
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut l = vec![vec![0.0; n]; n];
        for &(i, j, w) in &self.edges {
            l[i][i] += w;
            l[j][j] += w;
            l[i][j] -= w;
            l[j][i] -= w;
        }
        l
    }

    /// Find all oriented triangles (i,j,k) with i<j<k that form 3-cliques.
    /// Returns list of (i, j, k, edge_ij_idx, edge_jk_idx, edge_ik_idx) or None for missing edges.
    #[allow(clippy::type_complexity)]
    pub fn triangles(&self) -> Vec<(usize, usize, usize, Option<usize>, Option<usize>, Option<usize>)> {
        let mut tris = Vec::new();
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if self.adjacency[i][j] == 0.0 {
                    continue;
                }
                for k in (j + 1)..self.n {
                    if self.adjacency[i][k] > 0.0 && self.adjacency[j][k] > 0.0 {
                        let eij = self.edges.iter().position(|&(u, v, _)| u == i && v == j);
                        let ejk = self.edges.iter().position(|&(u, v, _)| u == j && v == k);
                        let eik = self.edges.iter().position(|&(u, v, _)| u == i && v == k);
                        tris.push((i, j, k, eij, ejk, eik));
                    }
                }
            }
        }
        tris
    }
}

/// Build a complete graph on `n` nodes with uniform weight `w`.
pub fn complete_graph(n: usize, w: f64) -> WeightedGraph {
    let mut g = WeightedGraph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            g.add_edge(i, j, w).unwrap();
        }
    }
    g
}
