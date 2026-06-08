use std::fmt;

/// Errors produced by the hodge-consensus crate.
#[derive(Debug)]
pub enum HodgeError {
    EdgeOutOfBounds { index: usize, num_edges: usize },
    NodeOutOfBounds { index: usize, n: usize },
    TooFewNodes { minimum: usize, actual: usize },
    FlowLengthMismatch { flow_len: usize, edge_count: usize },
    LabelCountMismatch { label_count: usize, node_count: usize },
    Underdetermined,
    SingularMatrix,
}

impl fmt::Display for HodgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EdgeOutOfBounds { index, num_edges } => {
                write!(f, "edge index {index} out of bounds (graph has {num_edges} edges)")
            }
            Self::NodeOutOfBounds { index, n } => {
                write!(f, "node index {index} out of bounds (graph has {n} nodes)")
            }
            Self::TooFewNodes { minimum, actual } => {
                write!(f, "graph must have at least {minimum} nodes, got {actual}")
            }
            Self::FlowLengthMismatch { flow_len, edge_count } => {
                write!(f, "flow vector length {flow_len} does not match edge count {edge_count}")
            }
            Self::LabelCountMismatch { label_count, node_count } => {
                write!(f, "label count {label_count} does not match node count {node_count}")
            }
            Self::Underdetermined => {
                write!(f, "underdetermined system: not enough comparisons to rank nodes")
            }
            Self::SingularMatrix => write!(f, "singular matrix in least-squares solve"),
        }
    }
}

impl std::error::Error for HodgeError {}
