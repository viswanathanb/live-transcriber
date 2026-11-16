//! Client for a whisper.cpp whisper-server (POST /inference).

use std::time::Duration;

use anyhow::{Result, bail};
use serde::Deserialize;

use crate::audio::SAMPLE_RATE;

#[derive(Debug, Clone, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(default)]
    pub no_speech_prob: f64,
    /// Word pieces with their own times; a piece without a leading space continues a word.
    #[serde(default)]
    pub words: Vec<Piece>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Piece {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    segments: Vec<Segment>,
    #[serde(default)]
    error: String,
}

pub struct Whisper {
    url: String,
    lang: String,
    agent: ureq::Agent,
}

/// Primes Japanese: without it the multilingual model leaves out 。 and 、.
fn prompt(lang: &str) -> Option<&'static str> {
    (lang == "ja").then_some("こんにちは。今日のニュースをお伝えします。")
}

impl Whisper {
    pub fn new(base: &str, lang: &str) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(60)))
            .build()
            .into();
        Whisper {
            url: format!("{}/inference", base.trim_end_matches('/')),
            lang: lang.to_string(),
            agent,
        }
    }

    pub fn transcribe(&self, pcm: &[i16]) -> Result<Vec<Segment>> {
        const BOUNDARY: &str = "----live-transcriber-boundary-7d1f";
        let mut body = Vec::with_capacity(pcm.len() * 2 + 1024);
        let mut field = |name: &str, value: &str| {
            body.extend_from_slice(
                format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes(),
            );
        };
        field("response_format", "verbose_json");
        field("temperature", "0.0");
        // No retries at higher temperatures: each one costs a full pass (seconds on music).
        field("temperature_inc", "0.0");
        field("language", &self.lang);
        if let Some(p) = prompt(&self.lang) {
            field("prompt", p);
        }
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"chunk.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        write_wav(&mut body, pcm);
        body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());

        let mut resp = self
            .agent
            .post(&self.url)
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={BOUNDARY}"),
            )
            .send(&body[..])?;
        let out: Response = serde_json::from_str(&resp.body_mut().read_to_string()?)?;
        if !out.error.is_empty() {
            bail!("whisper: {}", out.error);
        }
        Ok(out.segments)
    }
}

fn write_wav(w: &mut Vec<u8>, pcm: &[i16]) {
    let n = (pcm.len() * 2) as u32;
    let rate = SAMPLE_RATE as u32;
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + n).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&rate.to_le_bytes());
    w.extend_from_slice(&(rate * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&n.to_le_bytes());
    for s in pcm {
        w.extend_from_slice(&s.to_le_bytes());
    }
}
