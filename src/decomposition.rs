//! Hodge decomposition of opinion flows on a graph.
//!
//! The Hodge decomposition theorem states that any edge flow **f** on a graph
//! can be uniquely decomposed into three orthogonal components:
//!
//! **f = gradient + curl + harmonic**
//!
//! * **Gradient (exact):** globally consistent pairwise differences derivable
//!   from a scalar potential on the nodes.
//! * **Curl:** cyclic disagreements that form loops but average to zero.
//! * **Harmonic:** topological obstructions — disagreements that persist
//!   because of the graph's connected-component structure.
//!
//! This is the discrete analogue of the classical Hodge decomposition from
//! differential geometry.

use serde::{Deserialize, Serialize};

use crate::graph::OpinionGraph;

/// The three orthogonal components of a Hodge decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HodgeComponents {
    /// Gradient component: edge flows that are exact (derivable from a node potential).
    pub gradient: Vec<f64>,
    /// Curl component: edge flows that are co-exact (cyclic disagreements).
    pub curl: Vec<f64>,
    /// Harmonic component: edge flows that are harmonic (topological obstructions).
    pub harmonic: Vec<f64>,
    /// Original total flow (gradient + curl + harmonic = total).
    pub total: Vec<f64>,
}

impl HodgeComponents {
    /// Decompose the opinion flow of a graph into gradient + curl + harmonic.
    ///
    /// Uses an iterative least-squares projection approach:
    /// 1. Solve for the node potential that best approximates the gradient.
    /// 2. Compute residual.
    /// 3. Compute the curl via cyclic projections.
    /// 4. The remainder is harmonic.
    pub fn decompose(graph: &OpinionGraph) -> Self {
        let flow = graph.flow();
        let n = graph.n();
        let m = graph.m();
        let idx = graph.index_map();

        if m == 0 || n == 0 {
            return Self {
                gradient: vec![0.0; m],
                curl: vec![0.0; m],
                harmonic: vec![0.0; m],
                total: flow.clone(),
            };
        }

        // Step 1: Solve for gradient (potential on nodes).
        // Gradient flow: g_k = potential[i] - potential[j] for edge k = (i→j).
        // Least squares: B^T B φ = B^T f, where B is the incidence matrix.
        let lap = graph.laplacian();
        let bt_f = compute_bt_f(graph, &flow);
        let potential = solve_conjugate_gradient(&lap, &bt_f, 200, 1e-10);

        // Compute gradient component.
        let gradient: Vec<f64> = graph
            .edges
            .iter()
            .map(|(src, dst, _)| {
                let i = idx[src];
                let j = idx[dst];
                potential[i] - potential[j]
            })
            .collect();

        // Residual = total − gradient
        let residual: Vec<f64> = flow
            .iter()
            .zip(&gradient)
            .map(|(t, g)| t - g)
            .collect();

        // Step 2: Compute curl component via cycle projection.
        // Find simple cycles and project residual onto cycle space.
        let cycles = find_simple_cycles(graph);
        let curl = project_onto_cycles(&cycles, &residual, graph);

        // Step 3: Harmonic = residual − curl
        let harmonic: Vec<f64> = residual
            .iter()
            .zip(&curl)
            .map(|(r, c)| r - c)
            .collect();

        HodgeComponents {
            gradient,
            curl,
            harmonic,
            total: flow,
        }
    }

    /// Compute the norms of each component.
    pub fn norms(&self) -> ComponentNorms {
        ComponentNorms {
            gradient_norm: euclidean_norm(&self.gradient),
            curl_norm: euclidean_norm(&self.curl),
            harmonic_norm: euclidean_norm(&self.harmonic),
            total_norm: euclidean_norm(&self.total),
        }
    }

    /// Energy fractions: what proportion of the total disagreement energy is
    /// in each component.
    pub fn energy_fractions(&self) -> EnergyFractions {
        let norms = self.norms();
        let total = norms.total_norm;
        if total < 1e-15 {
            return EnergyFractions {
                gradient: 0.0,
                curl: 0.0,
                harmonic: 0.0,
            };
        }
        let t2 = total * total;
        EnergyFractions {
            gradient: (norms.gradient_norm * norms.gradient_norm) / t2,
            curl: (norms.curl_norm * norms.curl_norm) / t2,
            harmonic: (norms.harmonic_norm * norms.harmonic_norm) / t2,
        }
    }

    /// Verify orthogonality: dot products between components should be ≈ 0.
    pub fn verify_orthogonality(&self) -> OrthogonalityReport {
        let gc = dot(&self.gradient, &self.curl);
        let gh = dot(&self.gradient, &self.harmonic);
        let ch = dot(&self.curl, &self.harmonic);
        OrthogonalityReport {
            gradient_dot_curl: gc,
            gradient_dot_harmonic: gh,
            curl_dot_harmonic: ch,
            is_orthogonal: gc.abs() < 1e-6 && gh.abs() < 1e-6 && ch.abs() < 1e-6,
        }
    }

    /// Reconstruct the total flow from components (should equal `total`).
    pub fn reconstruct(&self) -> Vec<f64> {
        self.gradient
            .iter()
            .zip(&self.curl)
            .zip(&self.harmonic)
            .map(|((g, c), h)| g + c + h)
            .collect()
    }
}

/// Norms of the three Hodge components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNorms {
    pub gradient_norm: f64,
    pub curl_norm: f64,
    pub harmonic_norm: f64,
    pub total_norm: f64,
}

/// Fraction of total energy in each component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyFractions {
    pub gradient: f64,
    pub curl: f64,
    pub harmonic: f64,
}

/// Orthogonality verification between Hodge components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrthogonalityReport {
    pub gradient_dot_curl: f64,
    pub gradient_dot_harmonic: f64,
    pub curl_dot_harmonic: f64,
    pub is_orthogonal: bool,
}

// ---- Internal helpers ----

/// Compute B^T * f where B is the edge-node incidence matrix.
fn compute_bt_f(graph: &OpinionGraph, flow: &[f64]) -> Vec<f64> {
    let n = graph.n();
    let idx = graph.index_map();
    let mut result = vec![0.0; n];
    for (k, (src, dst, _)) in graph.edges.iter().enumerate() {
        let i = idx[src];
        let j = idx[dst];
        let s = graph.edges[k].2.abs().sqrt().max(0.0);
        result[i] += s * flow[k];
        result[j] -= s * flow[k];
    }
    result
}

/// Conjugate gradient solver for Lφ = b.
fn solve_conjugate_gradient(l: &[Vec<f64>], b: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);

    if rs_old < tol * tol {
        return x;
    }

    for _ in 0..max_iter {
        let ap = mat_vec(l, &p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rs_old / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot(&r, &r);
        if rs_new < tol * tol {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
    x
}

/// Find simple cycles in the directed graph using DFS.
fn find_simple_cycles(graph: &OpinionGraph) -> Vec<Vec<(usize, usize)>> {
    let n = graph.n();
    let idx = graph.index_map();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (src, dst, _) in &graph.edges {
        let i = idx[src];
        let j = idx[dst];
        adj[i].push(j);
    }

    let mut cycles = Vec::new();
    let mut visited = vec![false; n];
    let mut path = Vec::new();
    let mut in_path = vec![false; n];

    for start in 0..n {
        dfs_cycles(start, start, &adj, &mut path, &mut in_path, &mut visited, &mut cycles, 8);
    }

    // Deduplicate cycles (sort each cycle and dedup)
    let mut unique: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cycle in &cycles {
        let mut key: Vec<usize> = cycle.iter().flat_map(|(a, b)| [*a, *b]).collect();
        key.sort();
        if seen.insert(key) {
            unique.push(cycle.clone());
        }
    }

    unique
}

fn dfs_cycles(
    start: usize,
    node: usize,
    adj: &[Vec<usize>],
    path: &mut Vec<(usize, usize)>,
    in_path: &mut [bool],
    _visited: &mut [bool],
    cycles: &mut Vec<Vec<(usize, usize)>>,
    max_depth: usize,
) {
    if path.len() > max_depth {
        return;
    }
    for &next in &adj[node] {
        if next == start && !path.is_empty() {
            cycles.push(path.clone());
        } else if !in_path[next] {
            in_path[next] = true;
            path.push((node, next));
            dfs_cycles(start, next, adj, path, in_path, _visited, cycles, max_depth);
            path.pop();
            in_path[next] = false;
        }
    }
}

/// Project residual flow onto the cycle space to get the curl component.
fn project_onto_cycles(
    cycles: &[Vec<(usize, usize)>],
    residual: &[f64],
    graph: &OpinionGraph,
) -> Vec<f64> {
    let m = graph.m();
    let idx = graph.index_map();

    // Build edge index map: (src_idx, dst_idx) → edge index
    let mut edge_idx: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for (k, (src, dst, _)) in graph.edges.iter().enumerate() {
        let i = idx[src];
        let j = idx[dst];
        edge_idx.insert((i, j), k);
    }

    let mut curl = vec![0.0; m];

    if cycles.is_empty() {
        return curl;
    }

    // For each cycle, compute the projection coefficient and add contribution.
    for cycle in cycles {
        // Build the cycle indicator vector c (length m): c[k] = ±1 if edge k is in cycle.
        let mut c = vec![0.0; m];
        for &(i, j) in cycle {
            if let Some(&k) = edge_idx.get(&(i, j)) {
                c[k] += 1.0;
            }
            // Also consider reverse edge contribution
            if let Some(&k) = edge_idx.get(&(j, i)) {
                c[k] -= 1.0;
            }
        }

        let c_norm2 = dot(&c, &c);
        if c_norm2 < 1e-15 {
            continue;
        }
        let coeff = dot(residual, &c) / c_norm2;
        for k in 0..m {
            curl[k] += coeff * c[k];
        }
    }

    curl
}

fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| (0..n).map(|j| m[i][j] * v[j]).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn euclidean_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
