//! # hodge-consensus
//!
//! Hodge decomposition for agent disagreement resolution.
//!
//! When multiple agents assign scores or rankings to items, their pairwise disagreements
//! form a flow on a comparison graph. Hodge theory decomposes this flow into three
//! orthogonal components:
//!
//! - **Gradient flow** (`d₀(f)`): resolvable by adjusting node potentials — consensus is reachable.
//! - **Curl flow** (`δ₁(ω)`): cyclic disagreement around triangles — locally inconsistent.
//! - **Harmonic flow** (`h`): irreconcilable global disagreement — fundamental divergence.
//!
//! This lets you predict which disputes will naturally resolve and which represent
//! fundamental disagreements.
//!
//! ## Quick Start
//!
//! ```rust
//! use hodge_consensus::{WeightedGraph, EdgeFlow, ConsensusPredictor, RankAggregation};
//!
//! // Build a complete graph on 4 items
//! let graph = WeightedGraph::new(4);
//! // ... add edges and comparisons, then decompose
//! ```

pub mod consensus;
pub mod decomposition;
pub mod error;
pub mod flow;
pub mod graph;
pub mod ranking;

pub use consensus::ConsensusPredictor;
pub use decomposition::HodgeDecomposition;
pub use error::HodgeError;
pub use flow::EdgeFlow;
pub use graph::WeightedGraph;
pub use ranking::RankAggregation;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_complete_graph(n: usize) -> WeightedGraph {
        graph::complete_graph(n, 1.0)
    }

    // ── WeightedGraph tests ──

    #[test]
    fn test_graph_new_empty() {
        let g = WeightedGraph::new(3);
        assert_eq!(g.n, 3);
        assert_eq!(g.edges.len(), 0);
        assert_eq!(g.num_edges(), 0);
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 2.0).unwrap();
        g.add_edge(1, 2, 3.0).unwrap();
        assert_eq!(g.num_edges(), 2);
        assert_eq!(g.adjacency[0][1], 2.0);
        assert_eq!(g.adjacency[1][0], 2.0);
    }

    #[test]
    fn test_graph_add_edge_out_of_bounds() {
        let mut g = WeightedGraph::new(2);
        assert!(g.add_edge(0, 5, 1.0).is_err());
    }

    #[test]
    fn test_graph_laplacian() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(1, 2, 1.0).unwrap();
        let l = g.laplacian();
        assert_eq!(l[0][0], 1.0);
        assert_eq!(l[1][1], 2.0);
        assert_eq!(l[0][1], -1.0);
        assert_eq!(l[2][2], 1.0);
    }

    #[test]
    fn test_incidence_matrix() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 1, 4.0).unwrap();
        let b = g.incidence_matrix();
        assert_eq!(b.len(), 1);
        assert!((b[0][0] + 2.0).abs() < 1e-10);
        assert!((b[0][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_complete_graph() {
        let g = make_complete_graph(4);
        assert_eq!(g.num_edges(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_triangles() {
        let g = make_complete_graph(3);
        let tris = g.triangles();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0].0, 0);
        assert_eq!(tris[0].1, 1);
        assert_eq!(tris[0].2, 2);
    }

    // ── EdgeFlow tests ──

    #[test]
    fn test_edge_flow_new_valid() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(1, 2, 1.0).unwrap();
        let flow = EdgeFlow::new(g, vec![1.0, 2.0]);
        assert!(flow.is_ok());
        assert!((flow.unwrap().norm() - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_edge_flow_length_mismatch() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).unwrap();
        assert!(EdgeFlow::new(g, vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_edge_flow_from_comparisons() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(1, 2, 1.0).unwrap();
        g.add_edge(0, 2, 1.0).unwrap();
        let flow =
            EdgeFlow::from_comparisons(g, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]).unwrap();
        assert_eq!(flow.values.len(), 3);
    }

    // ── HodgeDecomposition tests ──

    #[test]
    fn test_pure_gradient_flow() {
        // Flow that is exactly the gradient of node potentials
        let g = make_complete_graph(3);
        // potentials: [0, 1, 3] → gradients: (0,1)=1, (0,2)=3, (1,2)=2
        let flow = EdgeFlow::new(g, vec![1.0, 3.0, 2.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();

        let g2: f64 = decomp.gradient_flow.iter().map(|x| x * x).sum();
        let c2: f64 = decomp.curl_flow.iter().map(|x| x * x).sum();
        let h2: f64 = decomp.harmonic_flow.iter().map(|x| x * x).sum();
        let total = g2 + c2 + h2;

        assert!(g2 / total > 0.99, "gradient should dominate");
    }

    #[test]
    fn test_pure_curl_flow() {
        // Cyclic flow on a triangle: (0,1)=1, (1,2)=1, (0,2)=-1
        // This sums to zero around the triangle but is not a gradient
        let g = make_complete_graph(3);
        let flow = EdgeFlow::new(g, vec![1.0, 1.0, -1.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();

        let (_gf, cf, _hf) = decomp.energy_fractions();
        // Curl should be significant for this cyclic flow
        assert!(cf > 0.01, "curl should be nonzero, got cf={}", cf);
    }

    #[test]
    fn test_zero_flow() {
        let g = make_complete_graph(3);
        let flow = EdgeFlow::new(g, vec![0.0, 0.0, 0.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();
        assert!(decomp.residual < 1e-10);
        assert!(decomp.gradient_flow.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_decomposition_residual_is_small() {
        let g = make_complete_graph(4);
        let flow = EdgeFlow::new(g, vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.7]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();
        assert!(
            decomp.residual < 1e-8,
            "residual should be near zero, got {}",
            decomp.residual
        );
    }

    #[test]
    fn test_energy_fractions_sum_to_one() {
        let g = make_complete_graph(4);
        let flow = EdgeFlow::new(g, vec![1.0, 2.0, 3.0, -1.0, 0.5, -2.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();
        let (gf, _cf, _hf) = decomp.energy_fractions();
        assert!((gf + _cf + _hf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_path_graph_decomposition() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).unwrap();
        g.add_edge(1, 2, 1.0).unwrap();
        g.add_edge(2, 3, 1.0).unwrap();
        // Pure gradient: potentials [0, 1, 2, 3]
        let flow = EdgeFlow::new(g, vec![1.0, 1.0, 1.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();
        let (gf, _, _) = decomp.energy_fractions();
        assert!(gf > 0.99, "path gradient should dominate");
    }

    // ── ConsensusPredictor tests ──

    #[test]
    fn test_consensus_all_resolvable() {
        let g = make_complete_graph(3);
        let flow = EdgeFlow::new(g, vec![1.0, 3.0, 2.0]).unwrap();
        let pred = ConsensusPredictor::new(&flow).unwrap();
        assert!(pred.predicts_consensus(0.8));
    }

    #[test]
    fn test_consensus_mixed_flow() {
        let g = make_complete_graph(4);
        let flow = EdgeFlow::new(g, vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
        let pred = ConsensusPredictor::new(&flow).unwrap();
        // Should have some classification
        assert_eq!(
            pred.resolvable_edges.len() + pred.irreconcilable_edges.len(),
            6
        );
    }

    #[test]
    fn test_consensus_zero_flow() {
        let g = make_complete_graph(3);
        let flow = EdgeFlow::new(g, vec![0.0, 0.0, 0.0]).unwrap();
        let pred = ConsensusPredictor::new(&flow).unwrap();
        assert!(pred.resolvable_edges.is_empty());
        assert!(pred.irreconcilable_edges.is_empty());
    }

    #[test]
    fn test_resolvable_fraction_range() {
        let g = make_complete_graph(4);
        let flow = EdgeFlow::new(g, vec![2.0, 1.0, 3.0, -0.5, 0.7, 1.2]).unwrap();
        let pred = ConsensusPredictor::new(&flow).unwrap();
        let f = pred.resolvable_fraction();
        assert!((0.0..=1.0).contains(&f));
    }

    // ── RankAggregation tests ──

    #[test]
    fn test_ranking_simple() {
        let g = make_complete_graph(3);
        let labels: Vec<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
        // B > A by 1, C > B by 1, C > A by 2 — consistent ranking A < B < C
        let rank = RankAggregation::rank(
            g,
            &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)],
            &labels,
        )
        .unwrap();
        assert_eq!(rank.nodes.len(), 3);
        assert!(rank.kendall_tau > 0.9, "should have high agreement");
    }

    #[test]
    fn test_ranking_order_correct() {
        let g = make_complete_graph(4);
        let labels: Vec<String> = ["D", "C", "B", "A"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // Consistent: A(3) > B(2) > C(1) > D(0)
        let rank = RankAggregation::rank(
            g,
            &[
                (0, 1, 1.0),
                (0, 2, 2.0),
                (0, 3, 3.0),
                (1, 2, 1.0),
                (1, 3, 2.0),
                (2, 3, 1.0),
            ],
            &labels,
        )
        .unwrap();
        let sorted = rank.sorted_indices();
        // Node 3 (A) should be highest, node 0 (D) lowest
        assert_eq!(sorted[0], 3);
        assert_eq!(sorted[3], 0);
    }

    #[test]
    fn test_ranking_label_mismatch() {
        let g = make_complete_graph(3);
        let labels: Vec<String> = ["A", "B"].iter().map(|s| s.to_string()).collect();
        assert!(RankAggregation::rank(g, &[], &labels).is_err());
    }

    #[test]
    fn test_ranking_kendall_tau_perfect() {
        let g = make_complete_graph(3);
        let labels: Vec<String> = ["X", "Y", "Z"].iter().map(|s| s.to_string()).collect();
        let rank = RankAggregation::rank(
            g,
            &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)],
            &labels,
        )
        .unwrap();
        assert!((rank.kendall_tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ranking_method_returns_labels() {
        let g = make_complete_graph(3);
        let labels: Vec<String> = ["A", "B", "C"].iter().map(|s| s.to_string()).collect();
        let rank = RankAggregation::rank(
            g,
            &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)],
            &labels,
        )
        .unwrap();
        let ranking = rank.ranking();
        assert_eq!(ranking.len(), 3);
    }

    // ── Integration tests ──

    #[test]
    fn test_4_node_mixed_decomposition() {
        let g = make_complete_graph(4);
        // Mixed flow with gradient + curl components
        let flow = EdgeFlow::new(g, vec![1.0, 2.0, 1.0, 0.5, -0.5, 0.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();

        // Reconstruction should be exact
        for k in 0..6 {
            let reconstructed = decomp.gradient_flow[k]
                + decomp.curl_flow[k]
                + decomp.harmonic_flow[k];
            assert!(
                (flow.values[k] - reconstructed).abs() < 1e-8,
                "edge {}: original={}, reconstructed={}",
                k,
                flow.values[k],
                reconstructed
            );
        }
    }

    #[test]
    fn test_large_graph() {
        let g = make_complete_graph(6);
        let m = g.num_edges();
        let values: Vec<f64> = (0..m).map(|i| i as f64 * 0.3 - 2.0).collect();
        let flow = EdgeFlow::new(g, values).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();
        assert!(decomp.residual < 1e-6);
    }

    #[test]
    fn test_weighted_graph_decomposition() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 2.0).unwrap();
        g.add_edge(1, 2, 3.0).unwrap();
        g.add_edge(0, 2, 1.0).unwrap();
        let flow = EdgeFlow::new(g, vec![1.0, 2.0, 3.0]).unwrap();
        let decomp = HodgeDecomposition::decompose(&flow).unwrap();
        assert!(decomp.residual < 1e-8);
    }

    #[test]
    fn test_serde_roundtrip() {
        let g = make_complete_graph(3);
        let flow = EdgeFlow::new(g, vec![1.0, 2.0, 0.5]).unwrap();
        let json = serde_json::to_string(&flow).unwrap();
        let flow2: EdgeFlow = serde_json::from_str(&json).unwrap();
        assert_eq!(flow2.values, flow.values);
        assert_eq!(flow2.graph.n, flow.graph.n);
    }
}
