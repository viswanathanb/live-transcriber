//! Audio input: ffmpeg decodes a stream URL or the microphone; Whisper gets 16 kHz mono PCM.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::thread;

use anyhow::{Context, Result};

use crate::player::Clock;

pub const SAMPLE_RATE: usize = 16_000;

pub enum Source {
    Url(String),
    /// avfoundation audio device index (`ffmpeg -f avfoundation -list_devices true -i ""`).
    Mic(String),
}

impl Source {
    pub fn label(&self) -> String {
        match self {
            Source::Url(u) => u.clone(),
            Source::Mic(d) => format!("microphone (avfoundation :{d})"),
        }
    }
}

pub struct Input {
    child: Child,
}

impl Drop for Input {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts ffmpeg and a reader thread that sends 100 ms blocks of 16 kHz mono
/// samples. The sender is dropped when the audio ends. With `play`, the audio
/// is decoded at the output's rate and channels, queued for playback, and
/// resampled for Whisper. `level` holds the RMS of the latest block (0..1, as
/// f32 bits) for the level meter; `received` counts the samples sent.
pub fn start(
    src: &Source,
    realtime: bool,
    play: Option<Arc<Clock>>,
    tx: Sender<Vec<i16>>,
    level: Arc<AtomicU32>,
    received: Arc<AtomicU64>,
) -> Result<Input> {
    let (rate, channels) = play
        .as_ref()
        .map_or((SAMPLE_RATE as u32, 1), |c| (c.rate, c.channels));
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
    match src {
        Source::Url(u) => {
            if u.starts_with("http://") || u.starts_with("https://") {
                cmd.args(["-user_agent", "live-transcriber/1.0"]);
                cmd.args([
                    "-reconnect",
                    "1",
                    "-reconnect_streamed",
                    "1",
                    "-reconnect_delay_max",
                    "5",
                ]);
            }
            // Start decoding at once instead of probing seconds of audio first.
            // (Not `-fflags nobuffer`: it throws the probed audio away.)
            cmd.args(["-probesize", "65536", "-analyzeduration", "500000"]);
            if realtime {
                cmd.arg("-re");
            }
            cmd.args(["-i", u]);
        }
        Source::Mic(dev) => {
            cmd.args(["-f", "avfoundation", "-i", &format!(":{dev}")]);
        }
    }
    cmd.args([
        "-vn",
        "-ac",
        &channels.to_string(),
        "-ar",
        &rate.to_string(),
    ]);
    cmd.args(["-f", "f32le", "pipe:1"]);
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("start ffmpeg (brew install ffmpeg)")?;
    let mut out = child.stdout.take().expect("piped stdout");

    thread::spawn(move || {
        let ch = channels as usize;
        let mut bytes = vec![0u8; rate as usize / 10 * ch * 4];
        let mut resampler = Resampler::new(rate as f64, SAMPLE_RATE as f64);
        let mut mono = Vec::new();
        let mut down = Vec::new();
        loop {
            if out.read_exact(&mut bytes).is_err() {
                return;
            }
            let (quads, _) = bytes.as_chunks::<4>();
            let samples: Vec<f32> = quads.iter().map(|&b| f32::from_le_bytes(b)).collect();
            if let Some(clock) = &play {
                clock.push(&samples);
            }
            mono.clear();
            mono.extend(
                samples
                    .chunks_exact(ch)
                    .map(|f| f.iter().sum::<f32>() / ch as f32),
            );
            down.clear();
            resampler.process(&mono, &mut down);
            let pcm: Vec<i16> = down
                .iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                .collect();
            level.store(rms(&pcm).to_bits(), Ordering::Relaxed);
            received.fetch_add(pcm.len() as u64, Ordering::Relaxed);
            if tx.send(pcm).is_err() {
                return;
            }
        }
    });
    Ok(Input { child })
}

/// Streaming windowed-sinc resampler (mono). It low-passes below the output's
/// Nyquist rate, so 44.1/48 kHz audio comes down to 16 kHz without aliasing.
struct Resampler {
    /// Input samples per output sample.
    step: f64,
    /// Input position of the next output sample, relative to `buf[0]`.
    pos: f64,
    buf: Vec<f32>,
    cutoff: f64,
}

/// Kernel half-width in input samples.
const HALF: usize = 24;

impl Resampler {
    fn new(from: f64, to: f64) -> Self {
        Resampler {
            step: from / to,
            pos: 0.0,
            buf: Vec::new(),
            cutoff: (to / from).min(1.0) * 0.9,
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.step == 1.0 {
            out.extend_from_slice(input);
            return;
        }
        self.buf.extend_from_slice(input);
        while (self.pos as usize) + HALF < self.buf.len() {
            let i0 = self.pos as isize;
            let (mut acc, mut wsum) = (0.0, 0.0);
            for j in (i0 - HALF as isize + 1).max(0)..=i0 + HALF as isize {
                let x = j as f64 - self.pos;
                let w = sinc(x * self.cutoff) * hann(x / (HALF as f64 + 1.0));
                acc += self.buf[j as usize] as f64 * w;
                wsum += w;
            }
            out.push((acc / wsum) as f32);
            self.pos += self.step;
        }
        let keep = (self.pos as usize).saturating_sub(HALF);
        self.buf.drain(..keep);
        self.pos -= keep as f64;
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let p = std::f64::consts::PI * x;
        p.sin() / p
    }
}

/// Hann window over -1..1.
fn hann(t: f64) -> f64 {
    if t.abs() >= 1.0 {
        0.0
    } else {
        0.5 + 0.5 * (std::f64::consts::PI * t).cos()
    }
}

/// Root mean square of the samples, 0..1.
pub fn rms(s: &[i16]) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let sum: f64 = s.iter().map(|&x| (x as f64) * (x as f64)).sum();
    ((sum / s.len() as f64).sqrt() / 32768.0) as f32
}

/// Prints the avfoundation devices ffmpeg can see.
pub fn list_devices() -> Result<()> {
    let out = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-f",
            "avfoundation",
            "-list_devices",
            "true",
            "-i",
            "",
        ])
        .output()
        .context("run ffmpeg")?;
    print!("{}", String::from_utf8_lossy(&out.stderr));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: f64, freq: f64, secs: f64) -> Vec<f32> {
        (0..(rate * secs) as usize)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / rate).sin() as f32 * 0.5)
            .collect()
    }

    fn level(s: &[f32]) -> f32 {
        (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt()
    }

    #[test]
    fn resamples_48k_to_16k_keeping_speech_and_cutting_alias() {
        for (freq, want) in [(1000.0, 0.35), (12000.0, 0.0)] {
            let mut r = Resampler::new(48000.0, 16000.0);
            let mut out = Vec::new();
            for block in tone(48000.0, freq, 1.0).chunks(4800) {
                r.process(block, &mut out);
            }
            assert!(
                (out.len() as i64 - 16000).abs() < 20,
                "{} samples",
                out.len()
            );
            let l = level(&out[100..]);
            assert!((l - want).abs() < 0.03, "{freq} Hz: level {l}");
        }
    }
}
