//! Agent ranking by agreeability.
//!
//! Each agent is scored by how much their opinion flow aligns with the gradient
//! (globally consistent) component. Agents with high agreeability are
//! cooperative; those with low agreeability are contrarians.

use serde::{Deserialize, Serialize};

use crate::decomposition::HodgeComponents;
use crate::graph::OpinionGraph;

/// Ranking of a single agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRanking {
    /// Agent name.
    pub agent: String,
    /// Overall agreeability score (0..1 range, higher = more agreeable).
    pub agreeability: f64,
    /// Alignment with the gradient component (−1..1).
    pub gradient_alignment: f64,
    /// The agent's rank (1 = most agreeable).
    pub rank: usize,
}

/// Complete ranking of all agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingReport {
    /// Agents ranked by agreeability (best first).
    pub rankings: Vec<AgentRanking>,
    /// Contrarians (bottom quartile by agreeability).
    pub contrarians: Vec<String>,
    /// Cooperators (top quartile by agreeability).
    pub cooperators: Vec<String>,
}

/// Rank all agents by agreeability.
pub fn rank_agents(graph: &OpinionGraph, decomp: &HodgeComponents) -> RankingReport {
    let idx = graph.index_map();
    let n = graph.n();

    let mut rankings: Vec<AgentRanking> = Vec::new();

    for (name, &i) in &idx {
        // Collect outgoing edge indices for this agent
        let mut total_flow = 0.0f64;
        let mut grad_flow = 0.0f64;
        let mut total_sq = 0.0f64;
        let mut grad_sq = 0.0f64;

        for (k, (src, _, _)) in graph.edges.iter().enumerate() {
            if src == name {
                total_flow += decomp.total[k];
                grad_flow += decomp.gradient[k];
                total_sq += decomp.total[k] * decomp.total[k];
                grad_sq += decomp.gradient[k] * decomp.gradient[k];
            }
        }

        let grad_norm = grad_sq.sqrt();
        let total_norm = total_sq.sqrt();

        let alignment = if total_norm > 1e-12 && grad_norm > 1e-12 {
            // Cosine similarity between agent's flow and gradient
            let dot: f64 = graph
                .edges
                .iter()
                .enumerate()
                .filter(|(_, (s, _, _))| s == name)
                .map(|(k, _)| decomp.total[k] * decomp.gradient[k])
                .sum();
            dot / (total_norm * grad_norm)
        } else if total_norm < 1e-12 {
            1.0 // No flow = trivially agreeable
        } else {
            0.0
        };

        let agreeability = (alignment + 1.0) / 2.0; // Map from [-1,1] to [0,1]

        rankings.push(AgentRanking {
            agent: name.clone(),
            agreeability,
            gradient_alignment: alignment,
            rank: 0,
        });
    }

    // Sort by agreeability descending
    rankings.sort_by(|a, b| b.agreeability.partial_cmp(&a.agreeability).unwrap());

    // Assign ranks
    for (i, r) in rankings.iter_mut().enumerate() {
        r.rank = i + 1;
    }

    let quartile = (n as f64 / 4.0).ceil() as usize;

    let cooperators: Vec<String> = rankings.iter().take(quartile).map(|r| r.agent.clone()).collect();
    let contrarians: Vec<String> = rankings.iter().rev().take(quartile).map(|r| r.agent.clone()).collect();

    RankingReport {
        rankings,
        contrarians,
        cooperators,
    }
}

/// Compute a single agent's agreeability.
pub fn agent_agreeability(
    graph: &OpinionGraph,
    decomp: &HodgeComponents,
    agent: &str,
) -> f64 {
    let idx = graph.index_map();
    if !idx.contains_key(agent) {
        return 0.0;
    }

    let mut total_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut grad_sq = 0.0f64;

    for (k, (src, _, _)) in graph.edges.iter().enumerate() {
        if src == agent {
            total_sq += decomp.total[k] * decomp.total[k];
            grad_sq += decomp.gradient[k] * decomp.gradient[k];
            dot += decomp.total[k] * decomp.gradient[k];
        }
    }

    let tn = total_sq.sqrt();
    let gn = grad_sq.sqrt();

    if tn < 1e-12 {
        return 1.0;
    }
    if gn < 1e-12 {
        return 0.5;
    }

    let alignment = dot / (tn * gn);
    (alignment + 1.0) / 2.0
}
