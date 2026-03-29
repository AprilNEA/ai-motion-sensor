use anyhow::{Context, Result};
use ort::session::Session;

/// Load an ONNX model from disk with CoreML acceleration when available.
pub fn load_model(path: &str) -> Result<Session> {
    tracing::info!(path, "loading ONNX model");

    let mut builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;

    // CoreML EP – best effort on macOS.
    #[cfg(target_os = "macos")]
    {
        builder = builder
            .with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }

    let session = builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("failed to load model: {path}"))?;

    tracing::info!(
        path,
        num_inputs = session.inputs().len(),
        num_outputs = session.outputs().len(),
        "model loaded"
    );
    Ok(session)
}
