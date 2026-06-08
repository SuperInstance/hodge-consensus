//! Directed weighted graph of agent opinions.
//!
//! An opinion graph models agents as nodes and agreement strengths as weighted
//! directed edges. The combinatorial Laplacian derived from this graph is the
//! central operator used in the Hodge decomposition.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A directed weighted graph of agent opinions.
///
/// Each node represents an agent (by name). Each directed edge `(src, dst, w)`
/// records that `src` agrees with `dst` at strength `w` (positive = agreement,
/// negative = disagreement).
/// A directed weighted graph of agent opinions.
///
/// Nodes are agents, edges are pairwise agreement strengths.
/// The combinatorial Laplacian L = D − A is the central operator for Hodge decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpinionGraph {
    /// Ordered list of unique agent names.
    pub agents: Vec<String>,
    /// Directed edges: (source, target, weight).
    pub edges: Vec<(String, String, f64)>,
}

impl OpinionGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Create a graph from an existing agent list (no edges).
    pub fn with_agents(agents: Vec<String>) -> Self {
        Self {
            agents,
            edges: Vec::new(),
        }
    }

    /// Add an agent. No-op if already present.
    pub fn add_agent(&mut self, name: &str) {
        if !self.agents.contains(&name.to_string()) {
            self.agents.push(name.to_string());
        }
    }

    /// Add a directed edge. Agents are added automatically if new.
    pub fn add_edge(&mut self, src: &str, dst: &str, weight: f64) {
        self.add_agent(src);
        self.add_agent(dst);
        self.edges.push((src.to_string(), dst.to_string(), weight));
    }

    /// Add a symmetric (bidirectional) edge with the same weight.
    pub fn add_symmetric_edge(&mut self, a: &str, b: &str, weight: f64) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }

    /// Number of agents (nodes).
    pub fn n(&self) -> usize {
        self.agents.len()
    }

    /// Number of edges.
    pub fn m(&self) -> usize {
        self.edges.len()
    }

    /// Build agent-name → index map.
    pub fn index_map(&self) -> HashMap<String, usize> {
        self.agents
            .iter()
            .enumerate()
            .map(|(i, a)| (a.clone(), i))
            .collect()
    }

    /// Weighted out-degree of agent `i`.
    pub fn out_degree(&self, idx: usize) -> f64 {
        let name = &self.agents[idx];
        self.edges
            .iter()
            .filter(|(s, _, _)| s == name)
            .map(|(_, _, w)| w.abs())
            .sum()
    }

    /// Weighted in-degree of agent `i`.
    pub fn in_degree(&self, idx: usize) -> f64 {
        let name = &self.agents[idx];
        self.edges
            .iter()
            .filter(|(_, d, _)| d == name)
            .map(|(_, _, w)| w.abs())
            .sum()
    }

    /// Compute the combinatorial Laplacian **L = D − A** (n×n).
    ///
    /// * `D` is the diagonal weighted out-degree matrix.
    /// * `A` is the weighted adjacency matrix (entry `A[i][j]` = weight of edge i→j).
    pub fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let mut l = vec![vec![0.0; n]; n];
        let idx = self.index_map();

        for (src, dst, w) in &self.edges {
            let i = idx[src];
            let j = idx[dst];
            l[i][j] -= w.abs();
            l[i][i] += w.abs();
        }
        l
    }

    /// Adjacency matrix (n×n). `A[i][j]` = absolute weight of edge i→j, or 0.
    pub fn adjacency(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let mut a = vec![vec![0.0; n]; n];
        let idx = self.index_map();
        for (src, dst, w) in &self.edges {
            let i = idx[src];
            let j = idx[dst];
            a[i][j] += w.abs();
        }
        a
    }

    /// Degree matrix (diagonal n×n).
    pub fn degree_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let mut d = vec![vec![0.0; n]; n];
        for i in 0..n {
            d[i][i] = self.out_degree(i);
        }
        d
    }

    /// Edge-incidence matrix **B** (m × n) for the directed edges.
    /// Row for edge k=(i→j): `B[k][i] = +√w`, `B[k][j] = −√w`.
    pub fn incidence(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let m = self.m();
        let mut b = vec![vec![0.0; n]; m];
        let idx = self.index_map();
        for (k, (src, dst, w)) in self.edges.iter().enumerate() {
            let i = idx[src];
            let j = idx[dst];
            let s = w.abs().sqrt().max(0.0);
            b[k][i] = s;
            b[k][j] = -s;
        }
        b
    }

    /// Opinion-flow vector (length m): one entry per edge.
    pub fn flow(&self) -> Vec<f64> {
        self.edges.iter().map(|(_, _, w)| *w).collect()
    }

    /// Build a complete graph with uniform weight (for testing / demos).
    pub fn complete(n: usize, weight: f64) -> Self {
        let agents: Vec<String> = (0..n).map(|i| format!("agent_{i}")).collect();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges.push((agents[i].clone(), agents[j].clone(), weight));
                }
            }
        }
        Self { agents, edges }
    }

    /// Build a ring graph: each agent agrees with the next (cyclic).
    pub fn ring(n: usize, weight: f64) -> Self {
        let agents: Vec<String> = (0..n).map(|i| format!("agent_{i}")).collect();
        let mut edges = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            edges.push((agents[i].clone(), agents[j].clone(), weight));
        }
        Self { agents, edges }
    }
}

impl Default for OpinionGraph {
    fn default() -> Self {
        Self::new()
    }
}
