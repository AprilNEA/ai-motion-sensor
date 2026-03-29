use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use ai_motion_sensor::config::AppConfig;
use ai_motion_sensor::pipeline::Pipeline;
use ai_motion_sensor::video::source::{
    FfmpegSource, FrameSource, ImageDirSource, RtspSource, is_rtsp_url,
};

/// AI-powered motion sensor with exit intent detection.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Input source: video file, image directory, or RTSP/RTSPS URL.
    ///
    /// Examples:
    ///   -i video.mp4
    ///   -i frames/
    ///   -i rtsps://admin:pass@192.168.1.100:554/stream
    #[arg(short, long)]
    input: String,

    /// Path to the configuration TOML file.
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    /// Override FPS (applies to all sources; auto-detected if omitted).
    #[arg(long)]
    fps: Option<f64>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Initialise ffmpeg (registers all codecs/formats).
    ffmpeg_next::init()?;

    let cli = Cli::parse();

    let config = AppConfig::load(&cli.config)
        .with_context(|| format!("failed to load config: {}", cli.config))?;

    // Create video source BEFORE pipeline to avoid potential fd interference
    // from ONNX Runtime model loading.
    let mut source: Box<dyn FrameSource> = if is_rtsp_url(&cli.input) {
        tracing::info!(url = %cli.input, "using RTSP stream source");
        Box::new(RtspSource::new(&cli.input)?)
    } else {
        let input_path = Path::new(&cli.input);
        if input_path.is_dir() {
            tracing::info!(dir = %input_path.display(), "using image directory source");
            Box::new(ImageDirSource::new(input_path, cli.fps.unwrap_or(30.0))?)
        } else {
            tracing::info!(file = %input_path.display(), "using ffmpeg video source");
            Box::new(FfmpegSource::new(&cli.input)?)
        }
    };

    let mut pipeline = Pipeline::new(config)?;
    pipeline.run(source.as_mut())?;

    tracing::info!("done");
    Ok(())
}
