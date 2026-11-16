//! Delayed playback on the Mac's default output. Audio is queued as it is
//! decoded and played `delay` seconds later, so Whisper has heard each word
//! before it plays; the playback position drives when the UI shows it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub struct Player {
    _stream: cpal::Stream,
    pub clock: Arc<Clock>,
}

/// The queue between the decoder and the output callback, and the playback position.
pub struct Clock {
    queue: Mutex<VecDeque<f32>>,
    /// Frames played or skipped: the position on the decoded audio's timeline.
    frames: AtomicU64,
    started: AtomicBool,
    muted: AtomicBool,
    pub rate: u32,
    pub channels: u16,
    delay: usize,
    pub delay_secs: f64,
}

impl Clock {
    /// Queues interleaved samples at `rate` / `channels`.
    pub fn push(&self, samples: &[f32]) {
        self.queue.lock().unwrap().extend(samples);
    }

    /// Seconds of the stream played so far.
    pub fn position(&self) -> f64 {
        self.frames.load(Ordering::Relaxed) as f64 / self.rate as f64
    }

    /// Seconds of audio queued and not yet played.
    pub fn buffered(&self) -> f64 {
        let n = self.queue.lock().unwrap().len();
        n as f64 / self.channels as f64 / self.rate as f64
    }

    pub fn started(&self) -> bool {
        self.started.load(Ordering::Relaxed)
    }

    pub fn muted(&self) -> bool {
        self.muted.load(Ordering::Relaxed)
    }

    pub fn toggle_mute(&self) {
        self.muted.fetch_xor(true, Ordering::Relaxed);
    }

    fn fill(&self, out: &mut [f32]) {
        let ch = self.channels as usize;
        let mut q = self.queue.lock().unwrap();
        let have = q.len() / ch;
        if !self.started() {
            if have < self.delay {
                out.fill(0.0);
                return;
            }
            self.started.store(true, Ordering::Relaxed);
        }
        // Keep the delay steady: if more than a second extra has queued up
        // (clock drift, a burst), skip ahead to it.
        if have > self.delay + self.rate as usize {
            let skip = have - self.delay;
            q.drain(..skip * ch);
            self.frames.fetch_add(skip as u64, Ordering::Relaxed);
        }
        let n = (out.len() / ch).min(q.len() / ch);
        let muted = self.muted();
        for (o, s) in out.iter_mut().zip(q.drain(..n * ch)) {
            *o = if muted { 0.0 } else { s };
        }
        out[n * ch..].fill(0.0);
        self.frames.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// Opens the default output device; playback starts once `delay` seconds are queued.
pub fn open(delay: f64) -> Result<Player> {
    let device = cpal::default_host()
        .default_output_device()
        .context("no audio output device")?;
    let config = device.default_output_config()?;
    if config.sample_format() != cpal::SampleFormat::F32 {
        bail!(
            "output device wants {:?} samples, not f32",
            config.sample_format()
        );
    }
    let rate = config.sample_rate();
    let clock = Arc::new(Clock {
        queue: Mutex::new(VecDeque::new()),
        frames: AtomicU64::new(0),
        started: AtomicBool::new(false),
        muted: AtomicBool::new(false),
        rate,
        channels: config.channels(),
        delay: (delay * rate as f64) as usize,
        delay_secs: delay,
    });
    let c = clock.clone();
    let stream = device.build_output_stream(
        config.config(),
        move |out: &mut [f32], _| c.fill(out),
        |_| {},
        None,
    )?;
    stream.play()?;
    Ok(Player {
        _stream: stream,
        clock,
    })
}
