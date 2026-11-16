//! Terminal UI: header with source and status, the transcript (paged: when it
//! reaches the bottom the screen clears and continues from the top), the live
//! partial line in gray, and key hints. With delayed playback, words
//! appear as they are played and the word being spoken is highlighted.

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{self, Event as Term, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::engine::{Event, Word, ends_sentence};
use crate::player::Clock;

/// Words after which a line without a sentence end is broken anyway (lyrics, run-ons).
const MAX_LINE: usize = 60;
/// Words show this much before they are played, like subtitles.
const LEAD: f64 = 0.1;

pub struct Info {
    pub title: String,
    pub source: String,
    pub lang: String,
    pub whisper: String,
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
    ended: bool,
    /// Lines scrolled up from the bottom of the whole transcript; 0 shows the live page.
    back: u16,
    /// Index of the first line on the live page.
    page: usize,
    saved: Option<String>,
}

/// Runs until q/Esc/Ctrl-C. Returns the path of the last saved transcript.
pub fn run(
    term: &mut DefaultTerminal,
    info: Info,
    events: Receiver<Event>,
    level: Arc<AtomicU32>,
    clock: Option<Arc<Clock>>,
    tick: Duration,
) -> Result<Option<String>> {
    let mut st = State {
        lines: Vec::new(),
        partial: Vec::new(),
        status: "Listening…".into(),
        error: None,
        took: None,
        pending: 0.0,
        lag: Duration::ZERO,
        upto: 0.0,
        ended: false,
        back: 0,
        page: 0,
        saved: None,
    };
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
                Event::Pass {
                    took,
                    pending,
                    lag,
                    upto,
                } => {
                    st.took = Some(took);
                    st.lag = lag;
                    st.pending = pending;
                    st.upto = upto;
                    st.status = "Transcribing".into();
                    st.error = None;
                }
                Event::Error(e) => st.error = Some(e),
                Event::Ended => {
                    st.ended = true;
                    st.status = "Stream ended".into();
                }
            }
        }
        let lvl = f32::from_bits(level.load(Ordering::Relaxed));
        term.draw(|f| draw(f, &info, &mut st, lvl, clock.as_deref()))?;

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

/// Cells in the buffer meter.
const METER: usize = 30;

/// The playback buffer against the delay it fills to: green has its text
/// already, yellow is waiting for Whisper, gray is not yet received.
fn buffer_meter(c: &Clock, ahead: f64) -> Line<'static> {
    let queued = c.buffered();
    let ready = ahead.clamp(0.0, queued);
    let cells = |s: f64| ((s / c.delay_secs * METER as f64).round() as usize).min(METER);
    let (full, done) = (cells(queued), cells(ready));
    Line::from(vec![
        Span::raw(" buffer "),
        Span::styled("█".repeat(done), Style::new().fg(Color::Green)),
        Span::styled("█".repeat(full - done), Style::new().fg(Color::Yellow)),
        Span::styled("░".repeat(METER - full), Style::new().fg(Color::DarkGray)),
        Span::raw(format!(
            " {queued:4.1} / {:.1} s   text ready {ready:4.1} s   ▶ {}",
            c.delay_secs,
            stamp(c.position())
        )),
    ])
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

fn draw(f: &mut ratatui::Frame, info: &Info, st: &mut State, level: f32, clock: Option<&Clock>) {
    let [head, body, foot] = Layout::vertical([
        Constraint::Length(if clock.is_some() { 4 } else { 3 }),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(f.area());

    // Without playback everything transcribed is shown at once.
    let pos = clock.map_or(f64::INFINITY, |c| c.position());
    let status = match clock {
        Some(c) if !c.started() && !st.ended => "Buffering…",
        Some(c) if c.muted() => "Muted",
        Some(_) if !st.ended => "Playing",
        _ => &st.status,
    };

    let bars = ((level * 400.0).sqrt() * 4.0).clamp(0.0, 20.0) as usize;
    let meter = format!("{:<20}", "▮".repeat(bars));
    let mut header = vec![
        Span::styled(
            format!(" {status} "),
            Style::new()
                .fg(Color::Black)
                .bg(if st.ended { Color::Gray } else { Color::Green }),
        ),
        Span::raw("  "),
        Span::styled(meter, Style::new().fg(Color::Green)),
        Span::raw(format!("  lang {}  ", info.lang)),
    ];
    if let Some(t) = st.took {
        header.push(Span::raw(format!(
            "lag {:.1}s  whisper {} ms  pending {:.1}s  ",
            st.lag.as_secs_f64(),
            t.as_millis(),
            st.pending
        )));
    }
    if let Some(e) = &st.error {
        header.push(Span::styled(format!("⚠ {e}"), Style::new().fg(Color::Red)));
    }
    let mut rows = vec![Line::from(header)];
    if let Some(c) = clock {
        rows.push(buffer_meter(c, st.upto - pos));
    }
    let header = Paragraph::new(rows).block(
        Block::new()
            .borders(Borders::ALL)
            .title(Line::from(vec![
                " ".into(),
                info.title.clone().bold(),
                " ".into(),
            ]))
            .title_bottom(Line::from(format!(" {} → {} ", info.source, info.whisper)).dark_gray()),
    );
    f.render_widget(header, head);

    let width = body.width.saturating_sub(2);
    let inner_h = body.height.saturating_sub(2);
    let wrapped = |text: Vec<Line<'static>>| Paragraph::new(text).wrap(Wrap { trim: false });
    let from = if st.back == 0 {
        // A full page starts over with the line being played on top.
        let (text, last) = transcript(st, st.page, pos);
        if wrapped(text).line_count(width) > inner_h as usize
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
