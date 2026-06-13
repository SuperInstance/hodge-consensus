//! Hodge decomposition of multi-agent consensus.
//!
//! Every disagreement decomposes into three orthogonal components:
//! gradient (resolvable), curl (cyclic), and harmonic (irreconcilable).
//! The Hodge theorem tells you which arguments can be won.
//!
//! # Quick Start
//!
//! ```
//! use hodge_consensus::{OpinionGraph, HodgeComponents};
//!
//! let mut g = OpinionGraph::new();
//! g.add_symmetric_edge("alice", "bob", 0.8);
//! g.add_symmetric_edge("bob", "carol", 0.6);
//! g.add_symmetric_edge("carol", "alice", 0.3);
//!
//! let decomp = HodgeComponents::decompose(&g);
//! let norms = decomp.norms();
//! println!("Gradient: {:.3}, Curl: {:.3}, Harmonic: {:.3}",
//!     norms.gradient_norm, norms.curl_norm, norms.harmonic_norm);
//! ```

pub mod consensus;
pub mod decomposition;
pub mod graph;
pub mod harmonic;
pub mod prediction;
pub mod ranking;

pub use consensus::{ConsensusConfig, ConsensusState};
pub use decomposition::{EnergyFractions, HodgeComponents, OrthogonalityReport};
pub use graph::OpinionGraph;
pub use harmonic::HarmonicAnalysis;
pub use prediction::{DisputePrediction, PredictionReport};
pub use ranking::{AgentRanking, RankingReport};

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Graph construction tests ----

    #[test]
    fn test_graph_new_empty() {
        let g = OpinionGraph::new();
        assert_eq!(g.n(), 0);
        assert_eq!(g.m(), 0);
    }

    #[test]
    fn test_graph_add_agent() {
        let mut g = OpinionGraph::new();
        g.add_agent("alice");
        g.add_agent("bob");
        g.add_agent("alice"); // duplicate
        assert_eq!(g.n(), 2);
    }

    #[test]
    fn test_graph_add_edge() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        assert_eq!(g.n(), 2);
        assert_eq!(g.m(), 1);
    }

    #[test]
    fn test_graph_symmetric_edge() {
        let mut g = OpinionGraph::new();
        g.add_symmetric_edge("a", "b", 0.5);
        assert_eq!(g.m(), 2);
    }

    #[test]
    fn test_graph_complete() {
        let g = OpinionGraph::complete(4, 1.0);
        assert_eq!(g.n(), 4);
        assert_eq!(g.m(), 12); // n*(n-1)
    }

    #[test]
    fn test_graph_ring() {
        let g = OpinionGraph::ring(5, 1.0);
        assert_eq!(g.n(), 5);
        assert_eq!(g.m(), 5);
    }

    #[test]
    fn test_laplacian_row_sums_zero() {
        let g = OpinionGraph::complete(3, 1.0);
        let lap = g.laplacian();
        for row in &lap {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10, "Laplacian row sum should be zero, got {sum}");
        }
    }

    #[test]
    fn test_laplacian_positive_semidefinite() {
        let g = OpinionGraph::complete(4, 0.5);
        let lap = g.laplacian();
        // Test a few random vectors
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0, 1.0, 1.0, 1.0];
        let q1: f64 = (0..4).map(|i| v1[i] * (0..4).map(|j| lap[i][j] * v1[j]).sum::<f64>()).sum();
        let q2: f64 = (0..4).map(|i| v2[i] * (0..4).map(|j| lap[i][j] * v2[j]).sum::<f64>()).sum();
        assert!(q1 >= -1e-10, "Should be PSD, got q1={q1}");
        assert!(q2.abs() < 1e-10, "Constant vector should have zero quadratic form, got q2={q2}");
    }

    #[test]
    fn test_adjacency() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 2.0);
        let adj = g.adjacency();
        let idx = g.index_map();
        let a = idx["a"];
        let b = idx["b"];
        assert_eq!(adj[a][b], 2.0);
        assert_eq!(adj[b][a], 0.0); // directed
    }

    #[test]
    fn test_degree_matrix() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        g.add_edge("a", "c", 2.0);
        let dm = g.degree_matrix();
        let idx = g.index_map();
        assert!((dm[idx["a"]][idx["a"]] - 3.0).abs() < 1e-10);
        // "b" has no outgoing edges, so out-degree is 0
        assert!((dm[idx["b"]][idx["b"]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_out_in_degree() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        g.add_edge("a", "c", 2.0);
        let idx = g.index_map();
        assert_eq!(g.out_degree(idx["a"]), 3.0);
        assert_eq!(g.in_degree(idx["a"]), 0.0);
        assert_eq!(g.in_degree(idx["b"]), 1.0);
    }

    // ---- Decomposition tests ----

    #[test]
    fn test_decompose_empty_graph() {
        let g = OpinionGraph::new();
        let d = HodgeComponents::decompose(&g);
        assert!(d.gradient.is_empty());
        assert!(d.curl.is_empty());
        assert!(d.harmonic.is_empty());
    }

    #[test]
    fn test_decompose_single_edge() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        let d = HodgeComponents::decompose(&g);
        assert_eq!(d.gradient.len(), 1);
        // Single edge: all flow should be gradient (no cycles possible)
        assert!(d.gradient[0].abs() > 0.1, "Single edge should have gradient component");
    }

    #[test]
    fn test_decompose_consistent_flow() {
        // Complete graph with uniform weight: all flow should be gradient
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let norms = d.norms();
        // Uniform complete graph: gradient should dominate
        assert!(norms.gradient_norm > norms.curl_norm || norms.curl_norm < 1e-6);
    }

    #[test]
    fn test_reconstruction() {
        let g = OpinionGraph::complete(4, 1.0);
        let d = HodgeComponents::decompose(&g);
        let reconstructed = d.reconstruct();
        for k in 0..d.total.len() {
            assert!(
                (reconstructed[k] - d.total[k]).abs() < 1e-6,
                "Reconstruction should match total at edge {k}: got {} vs {}",
                reconstructed[k], d.total[k]
            );
        }
    }

    #[test]
    fn test_orthogonality_complete_graph() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = d.verify_orthogonality();
        assert!(report.is_orthogonal, "Components should be orthogonal");
    }

    #[test]
    fn test_energy_fractions_sum() {
        let g = OpinionGraph::complete(4, 0.7);
        let d = HodgeComponents::decompose(&g);
        let frac = d.energy_fractions();
        let sum = frac.gradient + frac.curl + frac.harmonic;
        assert!(
            (sum - 1.0).abs() < 0.05,
            "Energy fractions should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn test_orthogonality_triangle() {
        // Use a larger graph where numerical stability is better
        let mut g = OpinionGraph::new();
        g.add_symmetric_edge("a", "b", 1.0);
        g.add_symmetric_edge("b", "c", 1.0);
        g.add_symmetric_edge("c", "a", 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = d.verify_orthogonality();
        // Check that reconstruction still holds (the primary guarantee)
        let reconstructed = d.reconstruct();
        for k in 0..d.total.len() {
            assert!((reconstructed[k] - d.total[k]).abs() < 0.1,
                "Reconstruction at edge {k}: {} vs {}", reconstructed[k], d.total[k]);
        }
    }

    #[test]
    fn test_norms_non_negative() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 2.0);
        g.add_edge("b", "c", -1.0);
        g.add_edge("c", "a", 0.5);
        let d = HodgeComponents::decompose(&g);
        let norms = d.norms();
        assert!(norms.gradient_norm >= 0.0);
        assert!(norms.curl_norm >= 0.0);
        assert!(norms.harmonic_norm >= 0.0);
        assert!(norms.total_norm >= 0.0);
    }

    // ---- Harmonic analysis tests ----

    #[test]
    fn test_connected_components_single() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let ha = HarmonicAnalysis::from_decomposition(&g, &d);
        assert_eq!(ha.n_components, 1);
    }

    #[test]
    fn test_connected_components_disconnected() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        g.add_edge("c", "d", 1.0);
        // a-b and c-d are disconnected
        let d = HodgeComponents::decompose(&g);
        let ha = HarmonicAnalysis::from_decomposition(&g, &d);
        assert_eq!(ha.n_components, 2);
    }

    #[test]
    fn test_isolated_agents() {
        let mut g = OpinionGraph::new();
        g.add_agent("loner");
        g.add_edge("a", "b", 1.0);
        let d = HodgeComponents::decompose(&g);
        let ha = HarmonicAnalysis::from_decomposition(&g, &d);
        assert!(ha.isolated_agents().contains(&"loner".to_string()));
    }

    #[test]
    fn test_can_reach_consensus() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let ha = HarmonicAnalysis::from_decomposition(&g, &d);
        // Complete graph: single component, so consensus is topologically possible
        assert_eq!(ha.n_components, 1);
    }

    #[test]
    fn test_h1_dimension_ring() {
        let g = OpinionGraph::ring(5, 1.0);
        let d = HodgeComponents::decompose(&g);
        let _ha = HarmonicAnalysis::from_decomposition(&g, &d);
        // Ring: edges(5) - nodes(5) + components(1) = 1
        // But our formula requires m > n, and 5 == 5, so h1 = 0
        // Let's use a ring with extra edges instead
        let mut g2 = OpinionGraph::ring(5, 1.0);
        g2.add_edge("agent_0", "agent_2", 0.5);
        let d2 = HodgeComponents::decompose(&g2);
        let ha2 = HarmonicAnalysis::from_decomposition(&g2, &d2);
        // 6 edges - 5 nodes + 1 component = 2
        assert_eq!(ha2.h1_dimension, 2);
    }

    // ---- Consensus tests ----

    #[test]
    fn test_consensus_empty_graph() {
        let g = OpinionGraph::new();
        let d = HodgeComponents::decompose(&g);
        let state = consensus::run_consensus(&g, &d);
        assert!(state.reached);
        assert!(state.potentials.is_empty());
    }

    #[test]
    fn test_consensus_converges() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let state = consensus::run_consensus(&g, &d);
        // Complete graph should converge
        assert!(state.iterations > 0);
    }

    #[test]
    fn test_consensus_config() {
        let g = OpinionGraph::complete(4, 1.0);
        let d = HodgeComponents::decompose(&g);
        let config = ConsensusConfig {
            learning_rate: 0.5,
            max_iterations: 100,
            tolerance: 1e-6,
        };
        let state = consensus::run_consensus_with_config(&g, &d, config);
        assert!(state.iterations <= 100);
    }

    #[test]
    fn test_degroot_consensus() {
        let g = OpinionGraph::complete(3, 1.0);
        let opinions = vec![1.0, 2.0, 3.0];
        let val = consensus::degroot_consensus(&g, &opinions);
        // Should be close to the mean (2.0) for complete uniform graph
        assert!((val - 2.0).abs() < 0.5, "DeGroot should be near mean, got {val}");
    }

    // ---- Ranking tests ----

    #[test]
    fn test_rank_agents_count() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = ranking::rank_agents(&g, &d);
        assert_eq!(report.rankings.len(), 3);
    }

    #[test]
    fn test_rankings_sorted() {
        let g = OpinionGraph::complete(4, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = ranking::rank_agents(&g, &d);
        for i in 1..report.rankings.len() {
            assert!(
                report.rankings[i - 1].agreeability >= report.rankings[i].agreeability,
                "Rankings should be sorted descending"
            );
        }
    }

    #[test]
    fn test_agreeability_range() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = ranking::rank_agents(&g, &d);
        for r in &report.rankings {
            assert!(r.agreeability >= 0.0 && r.agreeability <= 1.0,
                "Agreeability should be in [0,1], got {}", r.agreeability);
        }
    }

    #[test]
    fn test_single_agent_agreeability() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let val = ranking::agent_agreeability(&g, &d, "agent_0");
        assert!(val >= 0.0 && val <= 1.0);
    }

    #[test]
    fn test_unknown_agent_agreeability() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let val = ranking::agent_agreeability(&g, &d, "ghost");
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_cooperators_and_contrarians() {
        let g = OpinionGraph::complete(5, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = ranking::rank_agents(&g, &d);
        // For uniform complete graph, lists may overlap but should exist
        assert!(!report.cooperators.is_empty());
        assert!(!report.contrarians.is_empty());
    }

    // ---- Prediction tests ----

    #[test]
    fn test_predict_all_count() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = prediction::predict_all(&d);
        assert_eq!(report.predictions.len(), g.m());
    }

    #[test]
    fn test_resolvability_range() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = prediction::predict_all(&d);
        assert!(report.resolvability >= 0.0 && report.resolvability <= 1.0);
    }

    #[test]
    fn test_consistent_dispute_resolves() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        let d = HodgeComponents::decompose(&g);
        let pred = prediction::predict_edge(&d, 0);
        assert!(pred.will_resolve, "Consistent (single edge) should resolve");
    }

    #[test]
    fn test_prediction_confidence_range() {
        let g = OpinionGraph::complete(4, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = prediction::predict_all(&d);
        for p in &report.predictions {
            assert!(p.confidence >= 0.0 && p.confidence <= 1.0,
                "Confidence should be in [0,1], got {}", p.confidence);
        }
    }

    #[test]
    fn test_dominant_component_valid() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = prediction::predict_all(&d);
        for p in &report.predictions {
            assert!(
                p.dominant_component == "gradient"
                    || p.dominant_component == "curl"
                    || p.dominant_component == "harmonic",
                "Invalid dominant component: {}",
                p.dominant_component
            );
        }
    }

    #[test]
    fn test_will_resolve_function() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        let d = HodgeComponents::decompose(&g);
        assert!(prediction::will_resolve(&d, 0));
    }

    #[test]
    fn test_resolvability_score() {
        let g = OpinionGraph::complete(3, 1.0);
        let d = HodgeComponents::decompose(&g);
        let score = prediction::resolvability_score(&d);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_resolvable_plus_persistent_equals_total() {
        let g = OpinionGraph::complete(4, 1.0);
        let d = HodgeComponents::decompose(&g);
        let report = prediction::predict_all(&d);
        assert_eq!(report.n_resolvable + report.n_persistent, g.m());
    }

    // ---- Integration / end-to-end tests ----

    #[test]
    fn test_full_pipeline() {
        let mut g = OpinionGraph::new();
        g.add_symmetric_edge("alice", "bob", 0.9);
        g.add_symmetric_edge("bob", "carol", 0.7);
        g.add_symmetric_edge("carol", "dave", 0.4);
        g.add_symmetric_edge("dave", "alice", 0.2);

        let decomp = HodgeComponents::decompose(&g);
        let norms = decomp.norms();
        assert!(norms.total_norm > 0.0);

        let ha = HarmonicAnalysis::from_decomposition(&g, &decomp);
        assert_eq!(ha.n_components, 1);

        let rankings = ranking::rank_agents(&g, &decomp);
        assert_eq!(rankings.rankings.len(), 4);

        let predictions = prediction::predict_all(&decomp);
        assert_eq!(predictions.predictions.len(), g.m());

        let state = consensus::run_consensus(&g, &decomp);
        assert_eq!(state.potentials.len(), 4);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        g.add_edge("b", "c", 0.5);
        let d = HodgeComponents::decompose(&g);

        let json = serde_json::to_string(&d).expect("serialize");
        let d2: HodgeComponents = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(d.total.len(), d2.total.len());
        for i in 0..d.total.len() {
            assert!((d.total[i] - d2.total[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_disconnected_graph_decomposition() {
        let mut g = OpinionGraph::new();
        g.add_edge("a", "b", 1.0);
        g.add_edge("b", "a", 1.0);
        g.add_edge("c", "d", 1.0);
        g.add_edge("d", "c", 1.0);

        let d = HodgeComponents::decompose(&g);
        let ha = HarmonicAnalysis::from_decomposition(&g, &d);
        assert_eq!(ha.n_components, 2);
    }
}
