//! System load for the header: CPU and memory from sysinfo, GPU utilization and
//! GPU memory from the IOAccelerator's performance statistics (`ioreg`, no sudo).

use std::process::Command;
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::Duration;

use regex::Regex;
use sysinfo::System;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Load {
    /// All cores, 0..100.
    pub cpu: f32,
    pub gpu: Option<u32>,
    /// Bytes of memory the GPU has in use.
    pub gpu_mem: Option<u64>,
    pub mem_used: u64,
    pub mem_total: u64,
}

/// Samples the load every `every` in a background thread.
pub fn spawn(every: Duration) -> Arc<Mutex<Load>> {
    let load = Arc::new(Mutex::new(Load::default()));
    let out = load.clone();
    thread::spawn(move || {
        let Ok(mut sys) = System::new() else { return };
        loop {
            sys.refresh_cpu_usage();
            sys.refresh_memory();
            let (gpu, gpu_mem) = gpu();
            *out.lock().unwrap() = Load {
                cpu: sys.global_cpu_usage(),
                gpu,
                gpu_mem,
                mem_used: sys.used_memory(),
                mem_total: sys.total_memory(),
            };
            thread::sleep(every);
        }
    });
    load
}

static UTILIZATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""Device Utilization %"=(\d+)"#).unwrap());
static IN_USE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""In use system memory"=(\d+)"#).unwrap());

fn gpu() -> (Option<u32>, Option<u64>) {
    let Ok(out) = Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
    else {
        return (None, None);
    };
    parse_gpu(&String::from_utf8_lossy(&out.stdout))
}

/// The busiest GPU's utilization and the memory in use on all of them.
fn parse_gpu(ioreg: &str) -> (Option<u32>, Option<u64>) {
    let util = UTILIZATION
        .captures_iter(ioreg)
        .filter_map(|c| c[1].parse().ok())
        .max();
    let mem = IN_USE
        .captures_iter(ioreg)
        .filter_map(|c| c[1].parse::<u64>().ok())
        .reduce(|a, b| a + b);
    (util, mem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_gpu_performance_statistics() {
        let s = r#""PerformanceStatistics" = {"In use system memory (driver)"=0,"Device Utilization %"=67,"In use system memory"=2964897792,"Renderer Utilization %"=53}"#;
        assert_eq!(parse_gpu(s), (Some(67), Some(2_964_897_792)));
        assert_eq!(parse_gpu(""), (None, None));
    }
}
