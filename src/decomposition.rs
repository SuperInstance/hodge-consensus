use crate::error::HodgeError;
use crate::flow::EdgeFlow;
use serde::{Deserialize, Serialize};

/// Result of Hodge decomposition: flow = gradient + curl + harmonic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HodgeDecomposition {
    /// Gradient component: resolvable by adjusting node potentials.
    pub gradient_flow: Vec<f64>,
    /// Curl component: cyclic disagreement around triangles.
    pub curl_flow: Vec<f64>,
    /// Harmonic component: irreconcilable global disagreement.
    pub harmonic_flow: Vec<f64>,
    /// L2 residual ‖original − (gradient + curl + harmonic)‖.
    pub residual: f64,
}

impl HodgeDecomposition {
    /// Perform Hodge decomposition of an edge flow.
    ///
    /// 1. **Gradient**: project onto im(B) via least-squares solve of BᵀB f = Bᵀ ω
    /// 2. **Curl**: detect cyclic components from triangle constraints
    /// 3. **Harmonic**: remainder = original − gradient − curl
    pub fn decompose(flow: &EdgeFlow) -> Result<Self, HodgeError> {
        let m = flow.graph.edges.len();
        let n = flow.graph.n;
        let b = flow.graph.incidence_matrix();
        let lap = flow.graph.laplacian();

        // Step 1: Solve L f = Bᵀ ω for node potentials f
        let btw: Vec<f64> = (0..n)
            .map(|j| {
                b.iter()
                    .zip(flow.values.iter())
                    .map(|(row, &w)| row[j] * w)
                    .sum()
            })
            .collect();

        // Regularize: L + εI to handle disconnected components
        let potentials = solve_linear(&lap, &btw, 1e-10);

        // Gradient flow = B f
        let gradient_flow: Vec<f64> = b
            .iter()
            .map(|row| {
                row.iter()
                    .zip(potentials.iter())
                    .map(|(bij, fi)| bij * fi)
                    .sum()
            })
            .collect();

        // Step 2: Curl component from triangle constraints
        let triangles = flow.graph.triangles();
        let num_triangles = triangles.len();
        let mut curl: Vec<f64> = vec![0.0; m];

        if num_triangles > 0 {
            // Build curl matrix C (num_triangles × m): each triangle sums oriented edge flows
            let mut c_mat: Vec<Vec<f64>> = vec![vec![0.0; m]; num_triangles];
            for (t, (_i, _j, _k, eij, ejk, eik)) in triangles.iter().enumerate() {
                if let Some(idx) = eij {
                    c_mat[t][*idx] += 1.0;
                }
                if let Some(idx) = ejk {
                    c_mat[t][*idx] += 1.0;
                }
                if let Some(idx) = eik {
                    c_mat[t][*idx] -= 1.0;
                }
            }

            // Cᵀω
            let ct_omega: Vec<f64> = (0..num_triangles)
                .map(|t| {
                    c_mat[t]
                        .iter()
                        .zip(flow.values.iter())
                        .map(|(c, w)| c * w)
                        .sum()
                })
                .collect();

            // CᵀC
            let mut ctc: Vec<Vec<f64>> = vec![vec![0.0; num_triangles]; num_triangles];
            for i in 0..num_triangles {
                for j in 0..num_triangles {
                    ctc[i][j] = (0..m).map(|k| c_mat[i][k] * c_mat[j][k]).sum();
                }
            }

            let curl_coeffs = solve_linear(&ctc, &ct_omega, 1e-10);

            // curl_flow = Cᵀ curl_coeffs
            for k in 0..m {
                curl[k] = (0..num_triangles)
                    .map(|t| c_mat[t][k] * curl_coeffs[t])
                    .sum();
            }
        }

        // Step 3: Harmonic = remainder
        let harmonic_flow: Vec<f64> = flow
            .values
            .iter()
            .zip(gradient_flow.iter().zip(curl.iter()))
            .map(|(&orig, (&g, &c))| orig - g - c)
            .collect();

        let residual = flow
            .values
            .iter()
            .zip(gradient_flow.iter().zip(curl.iter().zip(harmonic_flow.iter())))
            .map(|(&orig, (&g, (&c, &h)))| {
                let d = orig - g - c - h;
                d * d
            })
            .sum::<f64>()
            .sqrt();

        Ok(Self {
            gradient_flow,
            curl_flow: curl,
            harmonic_flow,
            residual,
        })
    }

    /// Energy fractions: (gradient_frac, curl_frac, harmonic_frac).
    pub fn energy_fractions(&self) -> (f64, f64, f64) {
        let g2: f64 = self.gradient_flow.iter().map(|x| x * x).sum();
        let c2: f64 = self.curl_flow.iter().map(|x| x * x).sum();
        let h2: f64 = self.harmonic_flow.iter().map(|x| x * x).sum();
        let total = g2 + c2 + h2;
        if total < 1e-30 {
            (0.0, 0.0, 0.0)
        } else {
            (g2 / total, c2 / total, h2 / total)
        }
    }
}

/// Solve (A + εI) x = b via Gaussian elimination with partial pivoting.
pub(crate) fn solve_linear(a: &[Vec<f64>], b: &[f64], eps: f64) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }

    // Augmented matrix [A+εI | b]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n + 1];
            for j in 0..n {
                row[j] = if j < a[i].len() { a[i][j] } else { 0.0 };
            }
            row[i] += eps;
            row[n] = if i < b.len() { b[i] } else { 0.0 };
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            continue;
        }

        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            x[i] = 0.0;
            continue;
        }
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    x
}
