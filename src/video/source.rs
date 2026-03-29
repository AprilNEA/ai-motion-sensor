use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::{Path, PathBuf};

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
// FFmpeg in-process source (video file, RTSP/RTSPS, any URL ffmpeg supports)
// ---------------------------------------------------------------------------

use ffmpeg_next as ffmpeg;

/// Decode video from any source ffmpeg supports (file, RTSP, RTSPS, …)
/// entirely in-process — no subprocess pipes.
pub struct FfmpegSource {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::codec::decoder::Video,
    scaler: ffmpeg::software::scaling::Context,
    stream_idx: usize,
    fps: f64,
    width: u32,
    height: u32,
    // Reusable buffers to avoid per-frame allocation.
    decoded_frame: ffmpeg::frame::Video,
    rgb_frame: ffmpeg::frame::Video,
}

impl FfmpegSource {
    pub fn new(path: &str) -> Result<Self> {
        // Set RTSP options for streams.
        let mut opts = ffmpeg::Dictionary::new();
        if is_rtsp_url(path) {
            opts.set("rtsp_transport", "tcp");
            opts.set("fflags", "nobuffer");
        }

        let input = ffmpeg::format::input_with_dictionary(&path, opts)
            .with_context(|| format!("failed to open input: {path}"))?;

        // Find the best video stream.
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .context("no video stream found")?;
        let stream_idx = stream.index();

        let fps = f64::from(stream.avg_frame_rate());
        let fps = if fps > 0.0 { fps } else { 30.0 };

        // Open decoder.
        let codec_params = stream.parameters();
        let codec_id = codec_params.id();
        let decoder = ffmpeg::codec::Context::from_parameters(codec_params)?
            .decoder()
            .video()?;

        let width = decoder.width();
        let height = decoder.height();

        // Scaler: source pixel format → RGB24.
        let scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::RGB24,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )?;

        tracing::info!(
            path,
            width,
            height,
            fps,
            codec = ?codec_id,
            "video source opened"
        );

        Ok(Self {
            input,
            decoder,
            scaler,
            stream_idx,
            fps,
            width,
            height,
            decoded_frame: ffmpeg::frame::Video::empty(),
            rgb_frame: ffmpeg::frame::Video::empty(),
        })
    }
}

impl FrameSource for FfmpegSource {
    fn next_frame(&mut self) -> Result<Option<DynamicImage>> {
        // Feed packets to the decoder until we get a frame.
        loop {
            // Try to receive a decoded frame first (there might be buffered ones).
            if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                // Convert to RGB24.
                self.scaler.run(&self.decoded_frame, &mut self.rgb_frame)?;
                let data = self.rgb_frame.data(0);
                let stride = self.rgb_frame.stride(0);
                let w = self.width as usize;
                let h = self.height as usize;

                // Copy rows (stride may be wider than width*3).
                let mut rgb_buf = Vec::with_capacity(w * h * 3);
                for y in 0..h {
                    let row_start = y * stride;
                    rgb_buf.extend_from_slice(&data[row_start..row_start + w * 3]);
                }

                let img = image::RgbImage::from_raw(self.width, self.height, rgb_buf)
                    .context("failed to construct RGB image")?;
                return Ok(Some(DynamicImage::ImageRgb8(img)));
            }

            // Send the next packet from the video stream to the decoder.
            match self.input.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() == self.stream_idx {
                        self.decoder.send_packet(&packet)?;
                    }
                    // Packets from other streams (audio, etc.) are silently skipped.
                }
                None => {
                    // EOF – flush the decoder.
                    self.decoder.send_eof()?;
                    if self.decoder.receive_frame(&mut self.decoded_frame).is_ok() {
                        self.scaler.run(&self.decoded_frame, &mut self.rgb_frame)?;
                        let data = self.rgb_frame.data(0);
                        let stride = self.rgb_frame.stride(0);
                        let w = self.width as usize;
                        let h = self.height as usize;
                        let mut rgb_buf = Vec::with_capacity(w * h * 3);
                        for y in 0..h {
                            let row_start = y * stride;
                            rgb_buf.extend_from_slice(&data[row_start..row_start + w * 3]);
                        }
                        let img =
                            image::RgbImage::from_raw(self.width, self.height, rgb_buf)
                                .context("failed to construct RGB image")?;
                        return Ok(Some(DynamicImage::ImageRgb8(img)));
                    }
                    return Ok(None);
                }
            }
        }
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}

// ---------------------------------------------------------------------------
// RTSP wrapper with auto-reconnect
// ---------------------------------------------------------------------------

/// RTSP/RTSPS source with automatic reconnection on stream drop.
pub struct RtspSource {
    url: String,
    inner: Option<FfmpegSource>,
    fps: f64,
    max_retries: u32,
}

impl RtspSource {
    pub fn new(url: &str) -> Result<Self> {
        let inner = FfmpegSource::new(url)?;
        let fps = inner.fps();
        Ok(Self {
            url: url.to_string(),
            inner: Some(inner),
            fps,
            max_retries: 10,
        })
    }
}

impl FrameSource for RtspSource {
    fn next_frame(&mut self) -> Result<Option<DynamicImage>> {
        for attempt in 0..=self.max_retries {
            let source = match &mut self.inner {
                Some(s) => s,
                None => {
                    tracing::warn!(url = %self.url, attempt, "reconnecting to RTSP stream");
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    match FfmpegSource::new(&self.url) {
                        Ok(s) => {
                            self.inner = Some(s);
                            self.inner.as_mut().unwrap()
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "reconnect failed");
                            continue;
                        }
                    }
                }
            };

            match source.next_frame() {
                Ok(Some(frame)) => return Ok(Some(frame)),
                Ok(None) => {
                    // Stream ended / dropped.
                    self.inner = None;
                    if attempt < self.max_retries {
                        tracing::warn!(attempt = attempt + 1, "RTSP stream dropped");
                        continue;
                    }
                    tracing::error!("RTSP stream lost after {} retries", self.max_retries);
                    return Ok(None);
                }
                Err(e) => {
                    self.inner = None;
                    if attempt < self.max_retries {
                        tracing::warn!(attempt = attempt + 1, error = %e, "RTSP decode error");
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Ok(None)
    }

    fn fps(&self) -> f64 {
        self.fps
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return true if the input looks like an RTSP/RTSPS URL.
pub fn is_rtsp_url(input: &str) -> bool {
    let lower = input.to_ascii_lowercase();
    lower.starts_with("rtsp://") || lower.starts_with("rtsps://")
}
