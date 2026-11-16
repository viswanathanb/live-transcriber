//! Channel names instead of URLs: Sveriges Radio's channels are built in
//! (`p1`, `p4gbg`, `ekot`).

use anyhow::{Result, bail};

/// (id, name). Every SR channel streams at https://live1.sr.se/<id>-aac-320.
const SR: &[(&str, &str)] = &[
    ("p1", "P1"),
    ("p2", "P2"),
    ("p2sm", "P2 Språk och musik"),
    ("p3", "P3"),
    ("dingata", "P3 Din gata"),
    ("p4plus", "P4 Plus"),
    ("p4sth", "P4 Stockholm"),
    ("p4gbg", "P4 Göteborg"),
    ("p4malm", "P4 Malmöhus"),
    ("p4uppl", "P4 Uppland"),
    ("p4ostg", "P4 Östergötland"),
    ("p4vast", "P4 Väst"),
    ("p4dala", "P4 Dalarna"),
    ("p4gavl", "P4 Gävleborg"),
    ("p4hall", "P4 Halland"),
    ("p4vrml", "P4 Värmland"),
    ("p4blek", "P4 Blekinge"),
    ("ekotdirekt", "Ekot sänder direkt"),
    ("finska", "Sisuradio"),
    ("p6", "P6"),
];

/// Short names for the ids above.
const ALIASES: &[(&str, &str)] = &[
    ("p4", "p4sth"),
    ("ekot", "ekotdirekt"),
    ("sisuradio", "finska"),
];

pub struct Channel {
    pub name: String,
    pub url: String,
    /// Spoken language, when known.
    pub lang: Option<String>,
}

fn key(s: &str) -> String {
    s.to_lowercase()
        .replace('+', "plus")
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn sr(id: &str, name: &str) -> Channel {
    Channel {
        name: format!("Sveriges Radio {name}"),
        url: format!("https://live1.sr.se/{id}-aac-320"),
        lang: Some(if id == "finska" { "fi" } else { "sv" }.into()),
    }
}

/// Finds a channel by SR id or name ("p1", "P4 Göteborg").
pub fn resolve(query: &str) -> Result<Channel> {
    let k = key(query);
    let k = ALIASES
        .iter()
        .find(|(a, _)| *a == k)
        .map_or(k.clone(), |(_, id)| id.to_string());
    if let Some((id, name)) = SR
        .iter()
        .find(|(id, name)| *id == k || key(name) == k || key(&format!("sr{name}")) == k)
    {
        return Ok(sr(id, name));
    }
    bail!("unknown channel {query:?}; see --channels")
}

/// Prints the built-in channels.
pub fn list() {
    println!("Sveriges Radio:");
    for (id, name) in SR {
        let alias = ALIASES
            .iter()
            .find(|(_, a)| a == id)
            .map(|(a, _)| format!(" (or {a})"))
            .unwrap_or_default();
        println!("  {id:<11} {name}{alias}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_sr_ids_names_and_aliases() {
        assert_eq!(resolve("p1").unwrap().url, "https://live1.sr.se/p1-aac-320");
        assert_eq!(
            resolve("P4").unwrap().url,
            "https://live1.sr.se/p4sth-aac-320"
        );
        assert_eq!(
            resolve("P4 Göteborg").unwrap().url,
            "https://live1.sr.se/p4gbg-aac-320"
        );
        assert_eq!(
            resolve("p4+").unwrap().url,
            "https://live1.sr.se/p4plus-aac-320"
        );
        assert!(resolve("nope").is_err());
    }
}
