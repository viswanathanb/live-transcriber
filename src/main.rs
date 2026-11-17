//! live-transcriber: live transcript of the microphone, a radio channel or any
//! audio stream in the terminal. Audio is decoded with ffmpeg and sent to a
//! whisper.cpp whisper-server.

mod audio;
mod channels;
mod engine;
mod player;
mod stats;
mod ui;
mod whisper;

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;

use audio::Source;

#[derive(Parser)]
#[command(
    version,
    about = "Live transcript of the microphone or an audio stream (whisper.cpp whisper-server)"
)]
struct Args {
    /// Channel (p1, p3, p4gbg, ekot), stream URL, file, or `mic`.
    /// Default: the microphone.
    url: Option<String>,
    /// List the channels and exit.
    #[arg(long)]
    channels: bool,
    /// Transcribe the microphone instead (avfoundation audio device index).
    #[arg(long, num_args = 0..=1, default_missing_value = "0", value_name = "DEVICE")]
    mic: Option<String>,
    /// List microphone/audio devices and exit.
    #[arg(long)]
    list_devices: bool,
    /// whisper-server base URL.
    #[arg(long, env = "WHISPER_URL", default_value = "http://localhost:8178")]
    whisper: String,
    /// Spoken language (sv, ja, en, auto). Default: the station's language, else sv.
    #[arg(long, short)]
    lang: Option<String>,
    /// Least new audio (seconds) before the next Whisper pass; higher means less
    /// GPU load. Default: 0.3, or with playback what the delay leaves room for
    /// (delay − 5 s, at most 20 s).
    #[arg(long)]
    step: Option<f64>,
    /// Commit text after this many seconds even if no sentence has ended.
    #[arg(long, default_value_t = 6.0)]
    max_pending: f64,
    /// Read a file at playback speed, as if it were live.
    #[arg(long)]
    realtime: bool,
    /// Don't play the audio on the Mac; show the transcript as soon as it is ready.
    #[arg(long)]
    no_play: bool,
    /// Seconds the Mac's playback runs behind the stream, so the text is ready
    /// when each word plays. Longer delays transcribe in bigger chunks: less GPU
    /// load and more context for Whisper (10 s ≈ 15% GPU).
    #[arg(long, default_value_t = 20.0)]
    delay: f64,
    /// Silence threshold (RMS, 0..1).
    #[arg(long, default_value_t = 0.008)]
    silence: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.list_devices {
        return audio::list_devices();
    }
    if args.channels {
        channels::list();
        return Ok(());
    }

    let (source, title, station_lang) = match (&args.mic, args.url.as_deref()) {
        (Some(dev), _) => (Source::Mic(dev.clone()), "Microphone".into(), String::new()),
        (None, None | Some("mic")) => (Source::Mic("0".into()), "Microphone".into(), String::new()),
        (None, Some(u)) if u.contains("://") || std::path::Path::new(u).exists() => {
            (Source::Url(u.into()), u.into(), String::new())
        }
        (None, Some(name)) => {
            let c = channels::resolve(name)?;
            (Source::Url(c.url), c.name, c.lang.unwrap_or_default())
        }
    };
    let lang = args
        .lang
        .clone()
        .or_else(|| (!station_lang.is_empty()).then_some(station_lang))
        .unwrap_or_else(|| "sv".into());

    let (audio_tx, audio_rx) = mpsc::channel();
    let (ev_tx, ev_rx) = mpsc::channel();
    let level = Arc::new(AtomicU32::new(0));
    let received = Arc::new(AtomicU64::new(0));
    // The microphone isn't played back: it would echo.
    let player = match source {
        Source::Url(_) if !args.no_play => Some(player::open(args.delay.max(0.5))?),
        _ => None,
    };
    let clock = player.as_ref().map(|p| p.clock.clone());
    // Played audio is read at playback speed, or a file would decode ahead and be skipped.
    let realtime = args.realtime || player.is_some();
    let _input = audio::start(
        &source,
        realtime,
        clock.clone(),
        audio_tx,
        level.clone(),
        received.clone(),
    )?;
    // A word must be transcribed before it plays: a pass starts MARGIN seconds
    // before its oldest new audio plays, and takes about a second.
    let step = args
        .step
        .unwrap_or(match &clock {
            Some(c) => (c.delay_secs - engine::MARGIN).clamp(0.3, 20.0),
            None => 0.3,
        })
        .max(0.1);
    engine::spawn(
        whisper::Whisper::new(&args.whisper, &lang),
        engine::Config {
            step,
            max_pending: args.max_pending,
            silence: args.silence,
        },
        audio_rx,
        ev_tx,
    );

    let info = ui::Info {
        title,
        source: source.label(),
        lang,
        whisper: args.whisper.clone(),
        step,
    };
    let mut terminal = ratatui::init();
    let res = ui::run(
        &mut terminal,
        info,
        ev_rx,
        ui::Meters {
            level,
            received,
            load: stats::spawn(Duration::from_secs(1)),
        },
        clock,
        Duration::from_millis(50),
    );
    ratatui::restore();
    if let Ok(Some(path)) = &res {
        println!("Transcript saved to {path}");
    }
    res.map(|_| ())
}
