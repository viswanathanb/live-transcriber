//! Terminal UI: header with source and status, the transcript (paged: when it
//! fills 80% of the screen it clears and continues from the top), the live
//! partial line in gray, and key hints. With delayed playback, words
//! appear as they are played and the word being spoken is highlighted.

use std::fs;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event as Term, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Padding, Paragraph, Row, Table, Wrap};

use crate::audio::SAMPLE_RATE;
use crate::engine::{Event, MAX_BUFFER, Word, ends_sentence};
use crate::player::Clock;
use crate::stats::Load;

/// Words after which a line without a sentence end is broken anyway (lyrics, run-ons).
const MAX_LINE: usize = 60;
/// Words show this much before they are played, like subtitles.
const LEAD: f64 = 0.1;
/// Share of the transcript's rows a page fills before it starts over. The rest
/// leaves room for the line being played to grow without scrolling.
const PAGE_FILL: f64 = 0.8;

pub struct Info {
    pub title: String,
    pub source: String,
    pub lang: String,
    pub whisper: String,
    /// Seconds of new audio the engine waits for before a pass.
    pub step: f64,
}

/// Values the audio and stats threads update.
pub struct Meters {
    /// RMS of the latest audio block (0..1, as f32 bits).
    pub level: Arc<AtomicU32>,
    /// 16 kHz samples received from the audio input.
    pub received: Arc<AtomicU64>,
    pub load: Arc<Mutex<Load>>,
}

struct State {
    lines: Vec<Vec<Word>>,
    partial: Vec<Word>,
    status: String,
    error: Option<String>,
    took: Option<Duration>,
    pending: f64,
    lag: Duration,
    upto: f64,
    /// The latest window's committed context and new audio, in seconds.
    window: Option<(f64, f64)>,
    /// Whisper is working on `window`; when false the last one was done or quiet.
    busy: bool,
    quiet: bool,
    /// Where the latest window ends; audio after it fills the next one.
    next_from: f64,
    ended: bool,
    /// Lines scrolled up from the bottom of the whole transcript; 0 shows the live page.
    back: u16,
    /// Index of the first line on the live page.
    page: usize,
    saved: Option<String>,
}

impl State {
    fn new() -> Self {
        State {
            lines: Vec::new(),
            partial: Vec::new(),
            status: "Listening…".into(),
            error: None,
            took: None,
            pending: 0.0,
            lag: Duration::ZERO,
            upto: 0.0,
            window: None,
            busy: false,
            quiet: false,
            next_from: 0.0,
            ended: false,
            back: 0,
            page: 0,
            saved: None,
        }
    }
}

/// Runs until q/Esc/Ctrl-C. Returns the path of the last saved transcript.
pub fn run(
    term: &mut DefaultTerminal,
    info: Info,
    events: Receiver<Event>,
    meters: Meters,
    clock: Option<Arc<Clock>>,
    tick: Duration,
) -> Result<Option<String>> {
    let mut st = State::new();
    loop {
        while let Ok(ev) = events.try_recv() {
            match ev {
                Event::Commit(words) => {
                    // One line per sentence; an unfinished one continues on the same line.
                    for w in words {
                        match st.lines.last_mut() {
                            Some(line)
                                if line.len() < MAX_LINE
                                    && !line.last().is_some_and(|w| ends_sentence(&w.text)) =>
                            {
                                line.push(w);
                            }
                            _ => st.lines.push(vec![w]),
                        }
                    }
                    st.error = None;
                }
                Event::Partial(p) => st.partial = p,
                Event::Window {
                    context,
                    new,
                    upto,
                    quiet,
                } => {
                    st.window = Some((context, new));
                    st.busy = !quiet;
                    st.quiet = quiet;
                    st.next_from = upto;
                }
                Event::Pass {
                    took,
                    pending,
                    lag,
                    upto,
                } => {
                    st.busy = false;
                    st.took = Some(took);
                    st.lag = lag;
                    st.pending = pending;
                    st.upto = upto;
                    st.status = "Transcribing".into();
                    st.error = None;
                }
                Event::Error(e) => {
                    st.busy = false;
                    st.error = Some(e);
                }
                Event::Ended => {
                    st.busy = false;
                    st.ended = true;
                    st.status = "Stream ended".into();
                }
            }
        }
        let lvl = f32::from_bits(meters.level.load(Ordering::Relaxed));
        let heard = meters.received.load(Ordering::Relaxed) as f64 / SAMPLE_RATE as f64;
        let sys = *meters.load.lock().unwrap();
        term.draw(|f| draw(f, &info, &mut st, lvl, heard, sys, clock.as_deref()))?;

        if !event::poll(tick)? {
            continue;
        }
        let Term::Key(k) = event::read()? else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => break,
            KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break,
            KeyCode::Char('c') => {
                st.lines.clear();
                st.back = 0;
                st.page = 0;
            }
            KeyCode::Char('m') => {
                if let Some(c) = &clock {
                    c.toggle_mute();
                }
            }
            KeyCode::Char('s') => st.saved = save(&info, &st.lines).ok(),
            KeyCode::Up | KeyCode::Char('k') => st.back = st.back.saturating_add(1),
            KeyCode::Down | KeyCode::Char('j') => st.back = st.back.saturating_sub(1),
            KeyCode::PageUp => st.back = st.back.saturating_add(10),
            KeyCode::PageDown => st.back = st.back.saturating_sub(10),
            KeyCode::End | KeyCode::Char('G') => st.back = 0,
            _ => {}
        }
    }
    Ok(st.saved)
}

fn stamp(secs: f64) -> String {
    let s = secs as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

/// The words already played at `pos`, with the one being spoken highlighted.
fn spans(words: &[Word], pos: f64, base: Style) -> Vec<Span<'static>> {
    words
        .iter()
        .take_while(|w| w.start <= pos + LEAD)
        .map(|w| {
            let speaking = w.start <= pos + LEAD && pos < w.end;
            let style = if speaking {
                base.fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                base
            };
            Span::styled(w.text.clone(), style)
        })
        .collect()
}

/// Cells in each buffer bar.
const METER: usize = 30;
/// Cells in each system gauge.
const GAUGE: usize = 10;
/// Width of the system panel, borders included.
const SYSTEM_WIDTH: u16 = 34;

fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

fn colored(text: String, color: Color) -> Span<'static> {
    Span::styled(text, Style::new().fg(color))
}

/// A bar `width` cells wide: the parts, (amount, color), in order on `scale`;
/// the rest is gray.
fn bar(parts: &[(f64, Color)], scale: f64, width: usize) -> Line<'static> {
    let mut spans = Vec::new();
    let mut used = 0;
    for &(v, color) in parts {
        let n = ((v / scale * width as f64).round().max(0.0) as usize).min(width - used);
        spans.push(colored("█".repeat(n), color));
        used += n;
    }
    spans.push(Span::styled("░".repeat(width - used), dim()));
    Line::from(spans)
}

/// A gauge for a share 0..1: green, yellow above 60%, red above 85%.
fn gauge(share: f64) -> Line<'static> {
    let color = match share {
        s if s > 0.85 => Color::Red,
        s if s > 0.6 => Color::Yellow,
        _ => Color::Green,
    };
    bar(&[(share.clamp(0.0, 1.0), color)], 1.0, GAUGE)
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// The buffers, one row each: name, bar, amount, note.
fn buffer_rows(
    st: &State,
    heard: f64,
    step: f64,
    playback: Option<(&Clock, f64)>,
) -> Vec<Row<'static>> {
    let mut rows = Vec::new();
    // Green has its text already, yellow waits for Whisper, gray isn't received yet.
    if let Some((c, ahead)) = playback {
        let queued = c.buffered();
        let ready = ahead.clamp(0.0, queued);
        rows.push(Row::new(vec![
            Cell::from("playback"),
            Cell::from(bar(
                &[(ready, Color::Green), (queued - ready, Color::Yellow)],
                c.delay_secs,
                METER,
            )),
            Cell::from(format!("{queued:4.1} / {:.1} s", c.delay_secs)),
            Cell::from(Line::from(vec![
                colored(format!("{ready:.1} s"), Color::Green),
                " text ready".into(),
            ])),
        ]));
    }

    // Blue is the overlap with the previous window, magenta the new audio.
    let (context, new) = st.window.unwrap_or_default();
    let (state, color) = if st.busy {
        ("⟳ transcribing", Color::Yellow)
    } else if st.quiet {
        ("· quiet, skipped", Color::DarkGray)
    } else if st.window.is_some() {
        ("✓ done", Color::Green)
    } else {
        ("waiting", Color::DarkGray)
    };
    rows.push(Row::new(vec![
        Cell::from("window"),
        Cell::from(bar(
            &[(context, Color::Blue), (new, Color::Magenta)],
            MAX_BUFFER,
            METER,
        )),
        Cell::from(Line::from(vec![
            colored(format!("{context:4.1}"), Color::Blue),
            " + ".into(),
            colored(format!("{new:.1} s"), Color::Magenta),
        ])),
        Cell::from(colored(state.into(), color)),
    ]));

    // Audio received since the latest window was taken.
    let filled = (heard - st.next_from).max(0.0);
    rows.push(Row::new(vec![
        Cell::from("next"),
        Cell::from(bar(&[(filled.min(step), Color::Cyan)], step, METER)),
        Cell::from(Line::from(vec![
            colored(format!("{filled:4.1}"), Color::Cyan),
            format!(" / {step:.1} s").into(),
        ])),
        Cell::from(""),
    ]));
    rows
}

/// CPU, GPU and memory of the whole Mac.
fn system_rows(l: Load) -> Vec<Row<'static>> {
    let total = l.mem_total.max(1) as f64;
    let row = |name: &'static str, gauge: Line<'static>, value: String| {
        Row::new(vec![
            Cell::from(Span::styled(name, dim())),
            Cell::from(gauge),
            Cell::from(value),
        ])
    };
    vec![
        row("cpu", gauge(l.cpu as f64 / 100.0), format!("{:.0}%", l.cpu)),
        match l.gpu {
            Some(g) => row("gpu", gauge(g as f64 / 100.0), format!("{g}%")),
            None => row("gpu", bar(&[], 1.0, GAUGE), "n/a".into()),
        },
        row(
            "mem",
            gauge(l.mem_used as f64 / total),
            format!("{:.1}/{:.1} GB", gb(l.mem_used), gb(l.mem_total)),
        ),
        match l.gpu_mem {
            Some(m) => row("vram", gauge(m as f64 / total), format!("{:.1} GB", gb(m))),
            None => row("vram", bar(&[], 1.0, GAUGE), "n/a".into()),
        },
    ]
}

/// Status and Whisper timings as label / value pairs, two per row.
fn stream_rows(
    info: &Info,
    st: &State,
    status: &str,
    level: f32,
    clock: Option<&Clock>,
) -> Vec<Row<'static>> {
    let pair = |a: &'static str, av: Line<'static>, b: &'static str, bv: Line<'static>| {
        Row::new(vec![
            Cell::from(Span::styled(a, dim())),
            Cell::from(av),
            Cell::from(Span::styled(b, dim())),
            Cell::from(bv),
        ])
    };
    let badge = Span::styled(
        format!(" {status} "),
        Style::new()
            .fg(Color::Black)
            .bg(if st.ended { Color::Gray } else { Color::Green }),
    );
    let bars = ((level * 400.0).sqrt() * 4.0).clamp(0.0, 20.0) as usize;
    let input = colored("▮".repeat(bars), Color::Green);
    let or_dash = |v: Option<String>| Line::from(v.unwrap_or_else(|| "–".into()));
    let took = st.took.map(|t| format!("{} ms", t.as_millis()));
    let seen = st.took.is_some();
    let position = clock.map_or("live".into(), |c| stamp(c.position()));
    vec![
        pair("status", badge.into(), "lang", info.lang.clone().into()),
        pair("input", input.into(), "whisper", or_dash(took)),
        pair(
            "lag",
            or_dash(seen.then(|| format!("{:.1} s", st.lag.as_secs_f64()))),
            "pending",
            or_dash(seen.then(|| format!("{:.1} s", st.pending))),
        ),
        pair(
            "position",
            position.into(),
            "step",
            format!("{:.1} s", info.step).into(),
        ),
    ]
}

/// Lines from `from` on as played at `pos`, plus the partial line. Also returns
/// the index of the last line with anything shown yet.
fn transcript(st: &State, from: usize, pos: f64) -> (Vec<Line<'static>>, Option<usize>) {
    let mut text: Vec<Line> = Vec::new();
    let mut last = None;
    for (i, words) in st.lines.iter().enumerate().skip(from) {
        let mut shown = spans(words, pos, Style::new());
        let Some(first) = shown.first_mut() else {
            continue;
        };
        first.content = first.content.trim_start().to_string().into();
        let mut line = vec![Span::styled(
            format!("{}  ", stamp(words[0].start)),
            Style::new().fg(Color::DarkGray),
        )];
        line.extend(shown);
        text.push(Line::from(line));
        last = Some(i);
    }
    let mut partial = spans(
        &st.partial,
        pos,
        Style::new().fg(Color::Gray).add_modifier(Modifier::ITALIC),
    );
    if let Some(first) = partial.first_mut() {
        first.content = first.content.trim_start().to_string().into();
        let mut line = vec![Span::styled("  ▸  ", Style::new().fg(Color::Yellow))];
        line.extend(partial);
        text.push(Line::from(line));
    }
    (text, last)
}

fn draw(
    f: &mut ratatui::Frame,
    info: &Info,
    st: &mut State,
    level: f32,
    heard: f64,
    load: Load,
    clock: Option<&Clock>,
) {
    let buffers_h = if clock.is_some() { 5 } else { 4 };
    let [head, buffers, body, foot] = Layout::vertical([
        Constraint::Length(6),
        Constraint::Length(buffers_h),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(f.area());
    let [stream, system] =
        Layout::horizontal([Constraint::Min(40), Constraint::Length(SYSTEM_WIDTH)]).areas(head);

    // Without playback everything transcribed is shown at once.
    let pos = clock.map_or(f64::INFINITY, |c| c.position());
    let status = match clock {
        Some(c) if !c.started() && !st.ended => "Buffering…",
        Some(c) if c.muted() => "Muted",
        Some(_) if !st.ended => "Playing",
        _ => &st.status,
    };

    let bottom = match &st.error {
        Some(e) => Line::from(format!(" ⚠ {e} ")).red(),
        None => Line::from(format!(" {} → {} ", info.source, info.whisper)).dark_gray(),
    };
    let stream_table = Table::new(
        stream_rows(info, st, status, level, clock),
        [
            Constraint::Length(8),
            Constraint::Length(22),
            Constraint::Length(8),
            Constraint::Fill(1),
        ],
    )
    .column_spacing(1)
    .block(
        Block::new()
            .borders(Borders::ALL)
            .padding(Padding::horizontal(1))
            .title(Line::from(vec![
                " ".into(),
                info.title.clone().bold(),
                " ".into(),
            ]))
            .title_bottom(bottom),
    );
    f.render_widget(stream_table, stream);

    let system_table = Table::new(
        system_rows(load),
        [
            Constraint::Length(4),
            Constraint::Length(GAUGE as u16),
            Constraint::Fill(1),
        ],
    )
    .column_spacing(2)
    .block(
        Block::new()
            .borders(Borders::ALL)
            .padding(Padding::horizontal(1))
            .title(" System "),
    );
    f.render_widget(system_table, system);

    let playback = clock.map(|c| (c, st.upto - pos));
    let buffers_table = Table::new(
        buffer_rows(st, heard, info.step, playback),
        [
            Constraint::Length(8),
            Constraint::Length(METER as u16),
            Constraint::Length(14),
            Constraint::Fill(1),
        ],
    )
    .column_spacing(2)
    .block(
        Block::new()
            .borders(Borders::ALL)
            .padding(Padding::horizontal(1))
            .title(" Buffers "),
    );
    f.render_widget(buffers_table, buffers);

    let width = body.width.saturating_sub(2);
    let inner_h = body.height.saturating_sub(2);
    let wrapped = |text: Vec<Line<'static>>| Paragraph::new(text).wrap(Wrap { trim: false });
    let from = if st.back == 0 {
        // A full page starts over with the line being played on top.
        let full = ((inner_h as f64 * PAGE_FILL) as usize).max(1);
        let (text, last) = transcript(st, st.page, pos);
        if wrapped(text).line_count(width) >= full
            && let Some(last) = last
            && last > st.page
        {
            st.page = last;
        }
        st.page
    } else {
        0
    };
    let (mut text, _) = transcript(st, from, pos);
    if text.is_empty() {
        text.push(Line::from("Waiting for speech…".dark_gray()));
    }
    let block = Block::new().borders(Borders::ALL).title(" Transcript ");
    let para = wrapped(text);
    // Scrolling only happens within a page if a single line outgrows it, or in history.
    let total = para.line_count(width) as u16;
    let bottom = total.saturating_sub(inner_h);
    let top = bottom.saturating_sub(st.back);
    f.render_widget(para.block(block).scroll((top, 0)), body);

    let follow = if st.back == 0 {
        "live page"
    } else {
        "scrolled (End to follow)"
    };
    let mute = if clock.is_some() { "  m mute" } else { "" };
    let saved = st
        .saved
        .as_deref()
        .map(|p| format!("  saved {p}"))
        .unwrap_or_default();
    f.render_widget(
        Paragraph::new(format!(
            " q quit  s save  c clear{mute}  ↑↓/PgUp/PgDn scroll  — {follow}{saved}"
        ))
        .dark_gray(),
        foot,
    );
}

fn save(info: &Info, lines: &[Vec<Word>]) -> Result<String> {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let path = format!("transcript-{ts}.txt");
    let mut out = format!("{}\n{}\n\n", info.title, info.source);
    for words in lines {
        let text: String = words.iter().map(|w| w.text.as_str()).collect();
        out.push_str(&format!("[{}] {}\n", stamp(words[0].start), text.trim()));
    }
    fs::write(&path, out)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    #[test]
    fn header_panels_line_up() {
        let info = Info {
            title: "Microphone".into(),
            source: "microphone (avfoundation :0)".into(),
            lang: "sv".into(),
            whisper: "http://localhost:8178".into(),
            step: 0.3,
        };
        let mut st = State::new();
        st.took = Some(Duration::from_millis(790));
        st.lag = Duration::from_millis(800);
        st.pending = 10.2;
        st.window = Some((5.0, 15.2));
        st.busy = true;
        st.next_from = 20.0;
        let load = Load {
            cpu: 34.0,
            gpu: Some(47),
            gpu_mem: Some(2_900_000_000),
            mem_used: 19_200_000_000,
            mem_total: 48_000_000_000,
        };
        let mut term = Terminal::new(TestBackend::new(100, 16)).unwrap();
        term.draw(|f| draw(f, &info, &mut st, 0.01, 20.1, load, None))
            .unwrap();
        let screen: Vec<String> = term
            .backend()
            .buffer()
            .content()
            .chunks(100)
            .map(|row| row.iter().map(|c| c.symbol()).collect())
            .collect();
        println!("{}", screen.join("\n"));
        let at = |row: usize, text: &str| screen[row].find(text);
        assert!(at(1, "status").is_some() && at(1, "cpu").is_some());
        assert_eq!(at(2, "input"), at(1, "status"));
        assert_eq!(at(8, "next"), at(7, "window"));
        assert!(at(7, "transcribing").is_some());
        assert!(at(3, "19.2/48.0 GB").is_some());
    }
}
