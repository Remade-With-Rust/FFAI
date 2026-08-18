//! Text normalization: raw text → speakable words.
//!
//! v1 scope, stated honestly: integer expansion (0..=9999), which is what a
//! read-sentence corpus can exercise. Dates, currency, ordinals,
//! abbreviations and roman numerals are recorded gaps (mission plan M-T3
//! extends this stage; the Harvard corpus contains no digits at all, so
//! nothing here is corpus-tuned).

/// Expand digit runs into words, leaving everything else untouched.
/// `"route 66"` → `"route sixty six"`.
#[must_use]
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            // Collect the run as CHARACTERS, and parse only when the run is
            // short enough to need a number. Accumulating into a u64 first had
            // two defects, both on caller-supplied text:
            //
            //   * `n = n * 10 + d` overflows u64 past 19 digits. In debug that
            //     panics — a denial of service reachable from `synthesize`
            //     with an ID or a long phone number. In release, where this
            //     workspace does NOT set overflow-checks, it silently wraps and
            //     the speaker reads out a different number entirely.
            //   * the digit-by-digit branch recovered the digits with
            //     `n.to_string()`, which drops leading zeros: "0071234" was
            //     spoken as "seventy one thousand two hundred thirty four"
            //     shaped digits, with the zeros gone.
            //
            // Found by tests/miri_safe.rs. The proptest `.*` strategy in
            // tests/properties.rs never produced a 20-digit run.
            let mut run = String::new();
            run.push(c);
            while let Some(&d) = chars.peek() {
                if !d.is_ascii_digit() {
                    break;
                }
                run.push(d);
                chars.next();
            }
            if run.len() <= 4 {
                // At most "9999", so this always fits and always parses.
                let n: u64 = run.parse().unwrap_or(0);
                out.push_str(&number_to_words(n));
            } else {
                // Long digit runs (phone numbers, IDs, years past 9999) read
                // digit by digit — the least-wrong default until M-T3 — and now
                // read the ORIGINAL characters, so leading zeros survive.
                for d in run.chars() {
                    let v = u64::from(d.to_digit(10).unwrap_or(0));
                    out.push_str(&number_to_words(v));
                    out.push(' ');
                }
                out.pop();
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn number_to_words(n: u64) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    match n {
        0..=19 => ONES[n as usize].to_string(),
        20..=99 => {
            let t = TENS[(n / 10) as usize];
            if n.is_multiple_of(10) {
                t.to_string()
            } else {
                format!("{t} {}", ONES[(n % 10) as usize])
            }
        }
        100..=999 => {
            let h = format!("{} hundred", ONES[(n / 100) as usize]);
            if n.is_multiple_of(100) {
                h
            } else {
                format!("{h} {}", number_to_words(n % 100))
            }
        }
        1000..=9999 => {
            let t = format!("{} thousand", ONES[(n / 1000) as usize]);
            if n.is_multiple_of(1000) {
                t
            } else {
                format!("{t} {}", number_to_words(n % 1000))
            }
        }
        _ => n.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_integers_in_place() {
        assert_eq!(normalize("route 66 opens"), "route sixty six opens");
        assert_eq!(normalize("30 days"), "thirty days");
        assert_eq!(normalize("2026 items"), "two thousand twenty six items");
        assert_eq!(normalize("103"), "one hundred three");
        assert_eq!(normalize("no digits here."), "no digits here.");
    }
}
