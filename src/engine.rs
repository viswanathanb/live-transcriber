//! Streaming transcription. The pending audio is re-sent to Whisper as soon as
//! the previous pass returns, so text shows up quickly as a live partial line.
//! Sentences that have ended are committed and their audio dropped; the rest
//! stays pending so Whisper hears a cut-off sentence whole on the next pass.
//! Words carry their times on the stream's timeline, for delayed playback.
//!
//! The audio sent to Whisper is a sliding window: the last CONTEXT seconds of
//! audio that was already committed, then the new audio. Whisper hears the
//! lead-in to the new words, and words it hears again in the context are
//! dropped. The next window fills while Whisper works on this one.

use std::sync::LazyLock;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use regex::Regex;

use crate::audio::{SAMPLE_RATE, rms};
#[cfg(test)]
use crate::whisper::Piece;
use crate::whisper::{Segment, Whisper};

pub struct Config {
    /// Seconds of new audio between passes.
    pub step: f64,
    /// Pending audio longer than this is committed even without a sentence end.
    pub max_pending: f64,
    /// RMS (0..1) below which audio counts as silence.
    pub silence: f32,
}

/// A whole word; times are seconds from the start of the audio. `text` starts
/// with a space when it begins a new word (concatenate without separators).
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

pub enum Event {
    /// Final words.
    Commit(Vec<Word>),
    /// The not yet final rest, replacing the previous partial.
    Partial(Vec<Word>),
    /// A window of audio was taken: `context` seconds already committed, then
    /// `new` seconds, ending at `upto` on the stream's timeline. Unless it is
    /// `quiet` (skipped), it goes to Whisper now.
    Window {
        context: f64,
        new: f64,
        upto: f64,
        quiet: bool,
    },
    /// A Whisper pass finished. `lag` is how long ago the newest audio in it
    /// arrived; `upto` is where that audio ends on the stream's timeline.
    Pass {
        took: Duration,
        pending: f64,
        lag: Duration,
        upto: f64,
    },
    Error(String),
    Ended,
}

/// Hard cap: Whisper's window is 30 s.
pub const MAX_BUFFER: f64 = 28.0;
/// Trailing quiet this long ends an utterance (mic pauses, talk-radio breaks).
const PAUSE: f64 = 0.6;
/// Seconds of audio kept before the next word when audio is dropped.
const OVERLAP: f64 = 0.4;
/// Seconds of already committed audio at the start of each window.
const CONTEXT: f64 = 5.0;
/// With playback, a pass starts this many seconds before its oldest new audio plays.
pub const MARGIN: f64 = 5.0;

pub fn spawn(whisper: Whisper, cfg: Config, audio: Receiver<Vec<i16>>, ui: Sender<Event>) {
    thread::spawn(move || run(whisper, cfg, audio, ui));
}

fn secs(n: usize) -> f64 {
    n as f64 / SAMPLE_RATE as f64
}

fn samples(s: f64) -> usize {
    (s.max(0.0) * SAMPLE_RATE as f64) as usize
}

fn run(whisper: Whisper, cfg: Config, audio: Receiver<Vec<i16>>, ui: Sender<Event>) {
    let step = samples(cfg.step);
    let cap = samples(MAX_BUFFER);
    let mut buf: Vec<i16> = Vec::with_capacity(cap);
    let mut offset = 0.0; // seconds from the start of the audio to buf[0]
    let mut ctx = 0; // samples at the start of buf that are committed context
    let mut eof = false;
    let mut said: Vec<String> = Vec::new(); // the last committed words, normalised
    let mut newest = Instant::now();

    while !eof || buf.len() > ctx {
        // Wait for a step of new audio, then take everything that queued up
        // during the last pass (or a podcast's backlog) up to the cap, so
        // passes run back to back.
        let mut fresh = 0;
        while fresh < step && !eof {
            match audio.recv() {
                Ok(b) => {
                    fresh += b.len();
                    buf.extend_from_slice(&b);
                    newest = Instant::now();
                }
                Err(_) => eof = true,
            }
        }
        while buf.len() < cap && !eof {
            match audio.try_recv() {
                Ok(b) => {
                    buf.extend_from_slice(&b);
                    newest = Instant::now();
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => eof = true,
            }
        }

        // All quiet: nothing to say, and Whisper hallucinates on silence. The
        // last bit is kept in case a word is just starting.
        let quiet = loudest(&buf[ctx..]) < cfg.silence;
        if quiet || buf.len() - ctx >= SAMPLE_RATE / 2 || eof {
            let _ = ui.send(Event::Window {
                context: secs(ctx),
                new: secs(buf.len() - ctx),
                upto: offset + secs(buf.len()),
                quiet,
            });
        }
        if quiet {
            let keep = if eof { 0 } else { samples(OVERLAP) };
            let drop = buf.len().saturating_sub(keep);
            buf.drain(..drop);
            offset += secs(drop);
            ctx = ctx.saturating_sub(drop);
            let _ = ui.send(Event::Partial(Vec::new()));
            continue;
        }
        if buf.len() - ctx < SAMPLE_RATE / 2 && !eof {
            continue;
        }

        let t = Instant::now();
        let segs = match whisper.transcribe(&buf) {
            Ok(s) => s,
            Err(e) => {
                let _ = ui.send(Event::Error(e.to_string()));
                if buf.len() >= cap {
                    offset += secs(buf.len());
                    buf.clear();
                    ctx = 0;
                }
                thread::sleep(Duration::from_millis(500));
                continue;
            }
        };
        let dur = secs(buf.len());
        let pending = secs(buf.len() - ctx);
        let _ = ui.send(Event::Pass {
            took: t.elapsed(),
            pending,
            lag: newest.elapsed(),
            upto: offset + dur,
        });

        let tail = &buf[buf.len().saturating_sub(samples(PAUSE))..];
        let paused = pending > 1.5 && rms(tail) < cfg.silence;
        let done = eof || paused || buf.len() >= cap;

        if segs.is_empty() {
            // Nothing recognised yet: a word may just be starting, so keep the
            // audio unless it has gone on too long (music).
            let _ = ui.send(Event::Partial(Vec::new()));
            if done || pending > cfg.max_pending {
                offset += dur;
                buf.clear();
                ctx = 0;
            }
            continue;
        }

        let cut = if done {
            segs.len()
        } else {
            commit_point(&segs, dur, pending, cfg.max_pending)
        };
        // Words in the context were committed by an earlier pass.
        let from = offset + secs(ctx);
        let mut partial = new_words(&segs[cut..], offset, from);
        if cut > 0 {
            let mut words = new_words(&segs[..cut], offset, from);
            words.drain(..repeated(&said, &words));
            if !words.is_empty() {
                remember(&mut said, &words);
                let _ = ui.send(Event::Commit(words));
            }
            // The pending rest starts a little early: its start time can be
            // late, and words heard twice are stripped by `repeated`. Before
            // it, CONTEXT seconds of committed audio are kept.
            let edge = if cut == segs.len() {
                buf.len()
            } else {
                samples(segs[cut].start - OVERLAP)
            }
            .clamp(ctx, buf.len());
            let drop = edge.saturating_sub(samples(CONTEXT));
            buf.drain(..drop);
            offset += secs(drop);
            ctx = edge - drop;
        }
        partial.drain(..repeated(&said, &partial));
        let _ = ui.send(Event::Partial(partial));
    }
    let _ = ui.send(Event::Ended);
}

/// Number of segments that are final: up to the last sentence end that is not
/// right at the edge of the audio (it may still grow), never starting the
/// pending rest mid-word. Long run-ons commit all but the last segment.
/// `dur` is the whole window, `pending` the part after the context.
fn commit_point(segs: &[Segment], dur: f64, pending: f64, max_pending: f64) -> usize {
    let mut cut = 0;
    for (i, s) in segs.iter().enumerate() {
        if ends_sentence(&s.text) && s.end < dur - 0.3 {
            cut = i + 1;
        }
    }
    if cut == 0 && pending > max_pending && segs.len() > 1 {
        cut = segs.len() - 1;
    }
    while cut > 0 && cut < segs.len() && !starts_word(&segs[cut].text) {
        cut -= 1;
    }
    cut
}

/// Committed words kept for `strip_repeat`.
const SAID: usize = 30;

fn norm(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn remember(said: &mut Vec<String>, words: &[Word]) {
    said.extend(words.iter().map(|w| norm(&w.text)));
    let extra = said.len().saturating_sub(SAID);
    said.drain(..extra);
}

/// How many words at the start of `words` repeat the end of what was already
/// committed. Whisper's timestamps are loose and OVERLAP keeps audio from
/// before the cut, so the next pass usually hears the last word or two again.
fn repeated(said: &[String], words: &[Word]) -> usize {
    (1..=words.len().min(said.len()))
        .rev()
        .find(|&k| {
            said[said.len() - k..]
                .iter()
                .zip(&words[..k])
                .all(|(a, b)| *a == norm(&b.text))
        })
        .unwrap_or(0)
}

/// The words of the speech segments, at absolute times. Whisper's pieces are
/// joined into whole words (a piece without a leading space continues one).
fn words_of(segs: &[Segment], offset: f64) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    for s in segs {
        let Some(text) = clean(s) else { continue };
        if s.words.is_empty() {
            out.push(Word {
                start: offset + s.start.max(0.0),
                end: offset + s.end,
                text,
            });
            continue;
        }
        for p in &s.words {
            let t = p.word.replace('\n', " ");
            if t.trim().is_empty() {
                continue;
            }
            match out.last_mut() {
                Some(w) if !starts_word(&t) => {
                    w.text.push_str(&t);
                    w.end = offset + p.end;
                }
                _ => out.push(Word {
                    start: offset + p.start.max(0.0),
                    end: offset + p.end,
                    text: t,
                }),
            }
        }
    }
    out
}

/// The words of `segs` that are mostly after `from`, on the stream's timeline.
fn new_words(segs: &[Segment], offset: f64, from: f64) -> Vec<Word> {
    let mut words = words_of(segs, offset);
    words.retain(|w| (w.start + w.end) / 2.0 >= from);
    words
}

fn loudest(buf: &[i16]) -> f32 {
    buf.chunks(SAMPLE_RATE / 10).map(rms).fold(0.0, f32::max)
}

pub fn ends_sentence(t: &str) -> bool {
    let t = t.trim_end();
    [".", "?", "!", "…", "。", "？", "！"]
        .iter()
        .any(|e| t.ends_with(e))
}

fn starts_word(t: &str) -> bool {
    t.starts_with(' ')
        || t.chars()
            .next()
            .is_some_and(|c| matches!(c, '\u{3040}'..='\u{30ff}' | '\u{4e00}'..='\u{9fff}'))
}

/// Whisper's well-known hallucinations on silence or music (subtitle credits).
static HALLUCINATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)textning\.nu|svensktextning|undertext(er|ning)|amara\.org|^tack för att (ni|du) (tittade|lyssnade)|ご視聴ありがとうございました|チャンネル登録").unwrap()
});
/// Only sound tags ("[MUSIK]", "(skratt)", "♪").
static ONLY_TAGS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\s*(\[[^\]]*\]|\([^)]*\)|♪+|\*[^*]*\*))+\s*$").unwrap());

/// The segment's text, or None for non-speech. A leading space marks a new word.
fn clean(s: &Segment) -> Option<String> {
    let t = s.text.trim();
    if t.is_empty() || s.no_speech_prob > 0.8 || ONLY_TAGS.is_match(t) || HALLUCINATION.is_match(t)
    {
        return None;
    }
    Some(if s.text.starts_with([' ', '\n']) {
        format!(" {t}")
    } else {
        t.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            start,
            end,
            text: text.into(),
            no_speech_prob: 0.0,
            words: Vec::new(),
        }
    }

    #[test]
    fn commits_up_to_last_finished_sentence() {
        let segs = [
            seg(0.0, 2.0, " Hej."),
            seg(2.0, 4.0, " Hur mår"),
            seg(4.0, 5.0, " du"),
        ];
        assert_eq!(commit_point(&segs, 6.0, 6.0, 10.0), 1);
    }

    #[test]
    fn keeps_sentence_at_the_edge_pending() {
        let segs = [seg(0.0, 2.0, " Hej."), seg(2.0, 5.9, " Hur mår du?")];
        assert_eq!(commit_point(&segs, 6.0, 6.0, 10.0), 1);
    }

    #[test]
    fn context_does_not_count_as_pending() {
        let segs = [seg(0.0, 5.0, " och sedan"), seg(5.0, 11.0, " gick vi")];
        assert_eq!(commit_point(&segs, 12.0, 7.0, 10.0), 0);
        assert_eq!(commit_point(&segs, 12.0, 11.0, 10.0), 1);
    }

    #[test]
    fn drops_words_heard_in_the_context() {
        let segs = [seg(0.0, 4.8, " Hej."), seg(4.9, 6.0, " Hur")];
        let w = new_words(&segs, 10.0, 15.0);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].text, " Hur");
    }

    #[test]
    fn long_run_on_commits_all_but_last_word_group() {
        let segs = [
            seg(0.0, 5.0, " och sedan"),
            seg(5.0, 9.0, " gick vi"),
            seg(9.0, 11.0, "ut"),
        ];
        assert_eq!(commit_point(&segs, 12.0, 12.0, 10.0), 1);
    }

    fn words(text: &str) -> Vec<Word> {
        text.split_inclusive(' ')
            .filter(|w| !w.trim().is_empty())
            .map(|w| Word {
                start: 0.0,
                end: 0.0,
                text: format!(" {}", w.trim()),
            })
            .collect()
    }

    #[test]
    fn finds_words_already_committed() {
        let mut said = Vec::new();
        remember(&mut said, &words("Idag pratar vi om vädret i Stockholm."));
        assert_eq!(repeated(&said, &words("i Stockholm.")), 2);
        assert_eq!(repeated(&said, &words("i Stockholm. Det blir soligt")), 2);
        assert_eq!(repeated(&said, &words("Göteborg är fint")), 0);
        assert_eq!(repeated(&said, &words("Det blir soligt")), 0);
        remember(&mut said, &words("igår kväll, det"));
        assert_eq!(repeated(&said, &words("Det var den första")), 1);
    }

    #[test]
    fn joins_pieces_into_words() {
        let mut s = seg(1.0, 3.0, " Hej välkommen.");
        s.words = [
            (" Hej", 1.0, 1.2),
            (" väl", 1.3, 1.5),
            ("kommen", 1.5, 1.9),
            (".", 1.9, 2.0),
        ]
        .iter()
        .map(|&(w, start, end)| Piece {
            word: w.into(),
            start,
            end,
        })
        .collect();
        let w = words_of(&[s], 10.0);
        assert_eq!(w.len(), 2);
        assert_eq!(w[1].text, " välkommen.");
        assert_eq!((w[1].start, w[1].end), (11.3, 12.0));
    }

    #[test]
    fn drops_tags_and_hallucinations() {
        assert!(clean(&seg(0.0, 1.0, " [MUSIK]")).is_none());
        assert!(clean(&seg(0.0, 1.0, " Tack för att ni tittade!")).is_none());
        assert_eq!(clean(&seg(0.0, 1.0, " Hej")).as_deref(), Some(" Hej"));
    }
}
