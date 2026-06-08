use crate::decomposition::HodgeDecomposition;
use crate::error::HodgeError;
use crate::flow::EdgeFlow;
use serde::{Deserialize, Serialize};

/// Predicts which disagreements can reach consensus.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusPredictor {
    pub decomposition: HodgeDecomposition,
    /// Edge indices where gradient energy dominates (>50% of total).
    pub resolvable_edges: Vec<usize>,
    /// Edge indices where harmonic energy dominates (>50% of total).
    pub irreconcilable_edges: Vec<usize>,
}

impl ConsensusPredictor {
    /// Build a predictor from a decomposed edge flow.
    pub fn new(flow: &EdgeFlow) -> Result<Self, HodgeError> {
        let decomp = HodgeDecomposition::decompose(flow)?;
        let m = flow.values.len();

        let (g2, c2, h2) = (
            decomp.gradient_flow.iter().map(|x| x * x).sum::<f64>(),
            decomp.curl_flow.iter().map(|x| x * x).sum::<f64>(),
            decomp.harmonic_flow.iter().map(|x| x * x).sum::<f64>(),
        );

        let total = g2 + c2 + h2;

        let (resolvable_edges, irreconcilable_edges) = if total < 1e-30 {
            (vec![], vec![])
        } else {
            let mut resolvable = Vec::new();
            let mut irreconcilable = Vec::new();
            for k in 0..m {
                let gk = decomp.gradient_flow[k] * decomp.gradient_flow[k];
                let hk = decomp.harmonic_flow[k] * decomp.harmonic_flow[k];
                if gk > hk {
                    resolvable.push(k);
                } else if hk > gk {
                    irreconcilable.push(k);
                }
            }
            (resolvable, irreconcilable)
        };

        Ok(Self {
            decomposition: decomp,
            resolvable_edges,
            irreconcilable_edges,
        })
    }

    /// Fraction of total energy in the gradient (resolvable) subspace.
    pub fn resolvable_fraction(&self) -> f64 {
        self.decomposition.energy_fractions().0
    }

    /// Fraction of total energy in the harmonic (irreconcilable) subspace.
    pub fn irreconcilable_fraction(&self) -> f64 {
        self.decomposition.energy_fractions().2
    }

    /// Predict consensus: returns true if gradient fraction exceeds threshold (default 0.9).
    pub fn predicts_consensus(&self, threshold: f64) -> bool {
        self.resolvable_fraction() >= threshold
    }
}
