//! Text normalization: raw text → speakable words.
//!
//! v1 scope, stated honestly: integer expansion (0..=9999), which is what a
//! read-sentence corpus can exercise. Dates, currency, ordinals,
//! abbreviations and roman numerals are recorded gaps (mission plan M-T3
//! extends this stage; the Harvard corpus contains no digits at all, so
//! nothing here is corpus-tuned).

/// Expand digit runs into words, leaving everything else untouched.
/// `"route 66"` → `"route sixty six"`.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            let mut n: u64 = c as u64 - '0' as u64;
            let mut digits = 1usize;
            while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                n = n * 10 + d as u64;
                digits += 1;
                chars.next();
            }
            if digits <= 4 {
                out.push_str(&number_to_words(n));
            } else {
                // Long digit runs (phone numbers, years past 9999) read digit
                // by digit — the least-wrong default until M-T3.
                for d in n.to_string().chars() {
                    out.push_str(&number_to_words(d.to_digit(10).unwrap() as u64));
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
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] =
        ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    match n {
        0..=19 => ONES[n as usize].to_string(),
        20..=99 => {
            let t = TENS[(n / 10) as usize];
            if n % 10 == 0 { t.to_string() } else { format!("{t} {}", ONES[(n % 10) as usize]) }
        }
        100..=999 => {
            let h = format!("{} hundred", ONES[(n / 100) as usize]);
            if n % 100 == 0 { h } else { format!("{h} {}", number_to_words(n % 100)) }
        }
        1000..=9999 => {
            let t = format!("{} thousand", ONES[(n / 1000) as usize]);
            if n % 1000 == 0 { t } else { format!("{t} {}", number_to_words(n % 1000)) }
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
