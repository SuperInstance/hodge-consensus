//! Basic example: Hodge decomposition of a small opinion graph.
//!
//! Run with: cargo run --example basic

use hodge_consensus::{OpinionGraph, HodgeComponents, HarmonicAnalysis};

fn main() {
    // Build an opinion graph with 4 agents
    let mut graph = OpinionGraph::new();
    graph.add_symmetric_edge("alice", "bob", 0.9);
    graph.add_symmetric_edge("bob", "carol", 0.7);
    graph.add_symmetric_edge("carol", "dave", 0.4);
    graph.add_symmetric_edge("dave", "alice", 0.2);

    println!("Opinion Graph: {} agents, {} edges", graph.n(), graph.m());

    // Decompose the opinion flow into Hodge components
    let decomp = HodgeComponents::decompose(&graph);
    let norms = decomp.norms();

    println!("\nHodge Decomposition:");
    println!("  Gradient norm:  {:.4}", norms.gradient_norm);
    println!("  Curl norm:      {:.4}", norms.curl_norm);
    println!("  Harmonic norm:  {:.4}", norms.harmonic_norm);
    println!("  Total norm:     {:.4}", norms.total_norm);

    // Energy fractions
    let energy = decomp.energy_fractions();
    println!("\nEnergy Fractions:");
    println!("  Gradient: {:.1}%", energy.gradient * 100.0);
    println!("  Curl:     {:.1}%", energy.curl * 100.0);
    println!("  Harmonic: {:.1}%", energy.harmonic * 100.0);

    // Orthogonality check
    let ortho = decomp.verify_orthogonality();
    println!("\nOrthogonality: {} (g·c={:.6}, g·h={:.6}, c·h={:.6})",
        if ortho.is_orthogonal { "✓ PASS" } else { "✗ FAIL" },
        ortho.gradient_dot_curl,
        ortho.gradient_dot_harmonic,
        ortho.curl_dot_harmonic,
    );

    // Harmonic analysis
    let harmonic = HarmonicAnalysis::from_decomposition(&graph, &decomp);
    println!("\nHarmonic Analysis:");
    println!("  Connected components: {}", harmonic.n_components);
    println!("  H¹ dimension: {}", harmonic.h1_dimension);
    println!("  Can reach consensus: {}", harmonic.can_reach_consensus(0.1));
    println!("  Isolated agents: {:?}", harmonic.isolated_agents());
}
