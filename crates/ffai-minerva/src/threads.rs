//! The default thread count on a phone's unequal cores.

/// Threads worth running by default, or `None` to leave it to the pool: the cores with at least
/// half the fastest core's capacity (Linux and Android report it; elsewhere, and where the cores
/// are equal, `None`). A fork-join waits for its slowest member, so a slow core holds back the
/// fast ones: on a Pixel 9 Pro XL (1 + 3 + 4 cores) 4 threads beat 8 by 1.34x, 6 of 6 rounds.
#[must_use]
pub fn recommended() -> Option<usize> {
    let caps: Vec<u32> = (0..256)
        .map_while(|c| {
            let p = format!("/sys/devices/system/cpu/cpu{c}/cpu_capacity");
            std::fs::read_to_string(p).ok()?.trim().parse().ok()
        })
        .collect();
    from_capacities(&caps)
}

fn from_capacities(caps: &[u32]) -> Option<usize> {
    let max = *caps.iter().max()?;
    let fast = caps
        .iter()
        .filter(|&&c| 2 * u64::from(c) >= u64::from(max))
        .count();
    (fast < caps.len()).then_some(fast)
}

#[cfg(test)]
mod tests {
    use super::from_capacities;

    #[test]
    fn counts_the_fast_cores() {
        let pixel = [188, 188, 188, 188, 731, 731, 731, 1024];
        assert_eq!(from_capacities(&pixel), Some(4));
        assert_eq!(from_capacities(&[1024; 8]), None);
        assert_eq!(from_capacities(&[512, 512, 1024, 1024]), None);
        assert_eq!(from_capacities(&[]), None);
    }
}
