use anyhow::{Context, Result};
use image::DynamicImage;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Abstraction over different video input sources.
pub trait FrameSource {
    /// Get the next frame, or `None` if the stream is exhausted.
    fn next_frame(&mut self) -> Result<Option<DynamicImage>>;
    /// Nominal frames per second (for timestamp calculation).
    fn fps(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Image directory source (for testing / development)
// ---------------------------------------------------------------------------

/// Reads frames from a directory of image files, sorted by name.
pub struct ImageDirSource {
    paths: Vec<PathBuf>,
    index: usize,
    fps: f64,
}

impl ImageDirSource {
    pub fn new(dir: &Path, fps: f64) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("cannot read directory: {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("jpg" | "jpeg" | "png" | "bmp" | "webp")
                )
            })
            .collect();
        paths.sort();
        tracing::info!(count = paths.len(), "loaded image directory");
        Ok(Self {
            paths,
            index: 0,
            fps,
        })
    }
}

impl FrameSource for ImageDirSource {
    fn next_frame(&mut self) -> Result<Option<DynamicImage>> {
        if self.index >= self.paths.len() {
            return Ok(None);
        }
        let path = &self.paths[self.index];
        self.index += 1;
        let img = image::open(path)
            .with_context(|| format!("failed to load image: {}", path.display()))?;
        Ok(Some(img))
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}

// ---------------------------------------------------------------------------
// FFmpeg pipe source (video file → raw RGB frames)
// ---------------------------------------------------------------------------

/// Decodes a video file by spawning `ffmpeg` and reading raw RGB24 frames from
/// its stdout.  This avoids any Rust FFmpeg binding dependency.
pub struct FfmpegSource {
    pub(super) child: Child,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) fps: f64,
    pub(super) buf: Vec<u8>,
}

impl FfmpegSource {
    pub fn new(video_path: &str, width: u32, height: u32, fps: f64) -> Result<Self> {
        let child = Command::new("ffmpeg")
            .args([
                "-i",
                video_path,
                "-map", "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{width}x{height}"),
                "-r",
                &format!("{fps}"),
                "-loglevel",
                "error",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn ffmpeg – is it installed?")?;

        let frame_bytes = (width * height * 3) as usize;
        Ok(Self {
            child,
            width,
            height,
            fps,
            buf: vec![0u8; frame_bytes],
        })
    }
}

impl FrameSource for FfmpegSource {
    fn next_frame(&mut self) -> Result<Option<DynamicImage>> {
        let stdout = self
            .child
            .stdout
            .as_mut()
            .context("ffmpeg stdout not captured")?;

        match stdout.read_exact(&mut self.buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }

        let img = image::RgbImage::from_raw(self.width, self.height, self.buf.clone())
            .context("failed to construct image from raw bytes")?;
        Ok(Some(DynamicImage::ImageRgb8(img)))
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}

impl Drop for FfmpegSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

// ---------------------------------------------------------------------------
// RTSP / RTSPS stream source
// ---------------------------------------------------------------------------

/// Connects to an RTSP/RTSPS stream via ffmpeg.
///
/// Stream resolution and FPS are auto-probed with `ffprobe` so the caller does
/// not need to specify them manually.
pub struct RtspSource {
    url: String,
    child: Option<Child>,
    width: u32,
    height: u32,
    fps: f64,
    buf: Vec<u8>,
    max_retries: u32,
}

/// Stream metadata obtained from `ffprobe`.
#[derive(Debug)]
pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

impl RtspSource {
    pub fn new(url: &str, fps_override: Option<f64>) -> Result<Self> {
        let info = probe_stream(url)?;
        let fps = fps_override.unwrap_or(info.fps);
        tracing::info!(
            url,
            width = info.width,
            height = info.height,
            fps,
            "connecting to RTSP stream"
        );

        let child = Self::spawn_ffmpeg(url)?;
        let frame_bytes = (info.width * info.height * 3) as usize;

        Ok(Self {
            url: url.to_string(),
            child: Some(child),
            width: info.width,
            height: info.height,
            fps,
            buf: vec![0u8; frame_bytes],
            max_retries: 10,
        })
    }

    fn spawn_ffmpeg(url: &str) -> Result<Child> {
        Command::new("ffmpeg")
            .args([
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-i", url,
                "-map", "0:v:0",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-loglevel", "error",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn ffmpeg for RTSP stream")
    }

    /// Kill the current ffmpeg process and reconnect.
    fn reconnect(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        tracing::warn!(url = %self.url, "reconnecting to RTSP stream");
        std::thread::sleep(std::time::Duration::from_secs(2));
        self.child = Some(Self::spawn_ffmpeg(&self.url)?);
        Ok(())
    }
}

impl FrameSource for RtspSource {
    fn next_frame(&mut self) -> Result<Option<DynamicImage>> {
        for attempt in 0..=self.max_retries {
            let child = match &mut self.child {
                Some(c) => c,
                None => {
                    self.reconnect()?;
                    self.child.as_mut().unwrap()
                }
            };

            let stdout = match child.stdout.as_mut() {
                Some(s) => s,
                None => {
                    self.reconnect()?;
                    continue;
                }
            };

            match stdout.read_exact(&mut self.buf) {
                Ok(()) => {
                    let img = image::RgbImage::from_raw(
                        self.width, self.height, self.buf.clone(),
                    )
                    .context("failed to construct image from raw bytes")?;
                    return Ok(Some(DynamicImage::ImageRgb8(img)));
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    if attempt < self.max_retries {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max = self.max_retries,
                            "RTSP stream dropped, reconnecting"
                        );
                        self.reconnect()?;
                        continue;
                    }
                    tracing::error!("RTSP stream lost after {} retries", self.max_retries);
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(None)
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}

impl Drop for RtspSource {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
        }
    }
}

/// Return true if the input looks like an RTSP/RTSPS URL.
pub fn is_rtsp_url(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    lower.starts_with("rtsp://") || lower.starts_with("rtsps://")
}

/// Use `ffprobe` to discover the resolution and frame rate of a stream / file.
pub fn probe_stream(url: &str) -> Result<StreamInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-rtsp_transport", "tcp",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "csv=p=0",
            url,
        ])
        .output()
        .context("failed to run ffprobe – is ffmpeg installed?")?;

    anyhow::ensure!(
        output.status.success(),
        "ffprobe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').collect();
    anyhow::ensure!(
        parts.len() >= 3,
        "unexpected ffprobe output: {stdout}"
    );

    let width: u32 = parts[0].parse().context("bad width from ffprobe")?;
    let height: u32 = parts[1].parse().context("bad height from ffprobe")?;

    // r_frame_rate is a fraction like "30/1" or "30000/1001".
    let fps = parse_frame_rate(parts[2]).unwrap_or(30.0);

    Ok(StreamInfo { width, height, fps })
}

fn parse_frame_rate(s: &str) -> Option<f64> {
    let (num, den) = s.split_once('/')?;
    let n: f64 = num.trim().parse().ok()?;
    let d: f64 = den.trim().parse().ok()?;
    if d == 0.0 { return None; }
    Some(n / d)
}
