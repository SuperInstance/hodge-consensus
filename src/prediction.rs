//! Dispute prediction based on Hodge decomposition.
//!
//! A disagreement that is mostly **curl** will resolve (agents cycle around
//! but converge). A disagreement that is mostly **harmonic** will not resolve
//! (topological obstruction — disconnected groups).

use serde::{Deserialize, Serialize};

use crate::decomposition::HodgeComponents;

/// Prediction for a specific dispute (edge).
/// Prediction for a specific edge dispute: will it resolve, confidence, and dominant component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputePrediction {
    /// Whether the dispute will resolve.
    pub will_resolve: bool,
    /// Confidence of the prediction (0..1).
    pub confidence: f64,
    /// Dominant component: "gradient", "curl", or "harmonic".
    pub dominant_component: String,
    /// Edge index this prediction refers to.
    pub edge_index: usize,
}

/// Overall prediction report for all disputes.
/// Overall prediction report with per-edge predictions and aggregate resolvability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionReport {
    /// Per-edge predictions.
    pub predictions: Vec<DisputePrediction>,
    /// Overall resolvability score (fraction of disputes that will resolve).
    pub resolvability: f64,
    /// Number of disputes predicted to resolve.
    pub n_resolvable: usize,
    /// Number of disputes predicted to persist.
    pub n_persistent: usize,
}

/// Predict which disputes will resolve for each edge.
pub fn predict_all(decomp: &HodgeComponents) -> PredictionReport {
    let m = decomp.total.len();
    let mut predictions = Vec::with_capacity(m);

    for k in 0..m {
        predictions.push(predict_edge(decomp, k));
    }

    let n_resolvable = predictions.iter().filter(|p| p.will_resolve).count();
    let n_persistent = m - n_resolvable;
    let resolvability = if m > 0 {
        n_resolvable as f64 / m as f64
    } else {
        1.0
    };

    PredictionReport {
        predictions,
        resolvability,
        n_resolvable,
        n_persistent,
    }
}

/// Predict resolution for a single edge.
pub fn predict_edge(decomp: &HodgeComponents, edge_index: usize) -> DisputePrediction {
    let g = decomp.gradient[edge_index].abs();
    let c = decomp.curl[edge_index].abs();
    let h = decomp.harmonic[edge_index].abs();
    let total = g + c + h;

    if total < 1e-12 {
        return DisputePrediction {
            will_resolve: true,
            confidence: 1.0,
            dominant_component: "gradient".to_string(),
            edge_index,
        };
    }

    let g_frac = g / total;
    let c_frac = c / total;
    let h_frac = h / total;

    let (dominant, dom_frac) = if g_frac >= c_frac && g_frac >= h_frac {
        ("gradient", g_frac)
    } else if c_frac >= h_frac {
        ("curl", c_frac)
    } else {
        ("harmonic", h_frac)
    };

    // Resolution logic:
    // - Gradient: already resolved (globally consistent)
    // - Curl: will resolve (cyclic → converges)
    // - Harmonic: won't resolve (topological obstruction)
    let (will_resolve, confidence) = match dominant {
        "gradient" => (true, g_frac),
        "curl" => (true, 0.5 + 0.5 * c_frac), // Likely resolves, moderate confidence
        "harmonic" => (false, h_frac),
        _ => (true, 0.5),
    };

    DisputePrediction {
        will_resolve,
        confidence,
        dominant_component: dominant.to_string(),
        edge_index,
    }
}

/// Quick check: will a specific dispute resolve?
pub fn will_resolve(decomp: &HodgeComponents, edge_index: usize) -> bool {
    predict_edge(decomp, edge_index).will_resolve
}

/// Overall dispute resolvability score for the entire graph.
pub fn resolvability_score(decomp: &HodgeComponents) -> f64 {
    let report = predict_all(decomp);
    report.resolvability
}
