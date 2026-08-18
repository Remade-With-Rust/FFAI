use crate::engine::Task;

/// Unified error type across all `FFai` crates.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The engine exists in the registry but its implementation hasn't landed.
    /// Phase 0 ships stubs on purpose — see ROADMAP.md for when each goes live.
    #[error(
        "engine `{engine}` ({task}) is a stub in this build — implementation is \
         scheduled in ROADMAP.md; run `ffai engines` to see engine status"
    )]
    NotImplemented { task: Task, engine: String },

    #[error(
        "no engine named `{name}` registered for task `{task}` — run `ffai engines` to list available engines"
    )]
    UnknownEngine { task: Task, name: String },

    #[error("media error: {0}")]
    Media(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
