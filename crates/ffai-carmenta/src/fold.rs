//! Fold stray full-width forms to ASCII on Latin lines
//! (docs/plans/commercial-gaps.md, gap 12).
//!
//! The SVTR recognizer's charset is multilingual: it contains both `0` and the
//! full-width `０` (U+FF10), both `,` and `，` (U+FF0C). On a Latin line it
//! occasionally emits the full-width twin — `"$25，０00"`, `"12／31／2025"` —
//! which looks right to a person and is rejected, or worse silently dropped,
//! by any number or date parser downstream.
//!
//! In CJK text the full-width forms are correct and must stay. So the fold is
//! decided per LINE: a line with no CJK character at all is a Latin line, and
//! its full-width ASCII variants (U+FF01..=U+FF5E) and ideographic space are
//! mapped to their ASCII equivalents. A line with any CJK character is left
//! exactly as read.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Lines the fold has changed in this process — the census that says how
/// often it fires on a corpus before anyone decides it is harmless.
pub static FIRED: AtomicUsize = AtomicUsize::new(0);

/// Fold `line` in place when [`fold_latin_fullwidth`] would change it,
/// counting the firing. `FFAI_NO_FOLD` turns the fold off for A/B runs.
pub fn apply(line: &mut String) {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| std::env::var_os("FFAI_NO_FOLD").is_some()) {
        return;
    }
    if let Some(folded) = fold_latin_fullwidth(line) {
        *line = folded;
        FIRED.fetch_add(1, Ordering::Relaxed);
    }
}

/// True for characters that mark a line as CJK text: CJK ideographs, kana,
/// Hangul, and CJK punctuation (but NOT the full-width ASCII block, which is
/// what the fold targets).
const fn is_cjk(c: char) -> bool {
    matches!(c,
        // CJK symbols and punctuation, except U+3000 (the ideographic space),
        // which a Latin line can carry and the fold converts
        '\u{3001}'..='\u{303F}'
        | '\u{3040}'..='\u{30FF}' // hiragana, katakana
        | '\u{3100}'..='\u{312F}' // bopomofo
        | '\u{3400}'..='\u{4DBF}' // CJK extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
        | '\u{AC00}'..='\u{D7AF}' // Hangul syllables
        | '\u{F900}'..='\u{FAFF}' // CJK compatibility ideographs
        | '\u{FF65}'..='\u{FFDC}' // half-width katakana, Hangul
    )
}

/// The line with full-width ASCII variants folded, when it contains no CJK
/// character; otherwise the line unchanged. `None` when nothing would change,
/// so a caller can count how often the fold fires.
#[must_use]
pub fn fold_latin_fullwidth(line: &str) -> Option<String> {
    if line.chars().any(is_cjk) {
        return None;
    }
    if !line
        .chars()
        .any(|c| matches!(c, '\u{FF01}'..='\u{FF5E}' | '\u{3000}'))
    {
        return None;
    }
    Some(
        line.chars()
            .map(|c| match c {
                // U+FF01..U+FF5E are U+0021..U+007E shifted by 0xFEE0.
                '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
                '\u{3000}' => ' ',
                other => other,
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::fold_latin_fullwidth as fold;

    #[test]
    fn stray_full_width_forms_on_a_latin_line_become_ascii() {
        assert_eq!(fold("$25，０00").as_deref(), Some("$25,000"));
        assert_eq!(fold("12／31／2025").as_deref(), Some("12/31/2025"));
        assert_eq!(fold("Ａmount：１０").as_deref(), Some("Amount:10"));
    }

    #[test]
    fn a_line_with_cjk_text_is_left_exactly_as_read() {
        assert_eq!(fold("金额：１０，０００元"), None);
        assert_eq!(fold("東京（ＴＯＫＹＯ）"), None);
        assert_eq!(fold("서울，２０２５"), None);
    }

    #[test]
    fn a_plain_line_is_not_touched() {
        assert_eq!(fold("$25,000"), None);
        assert_eq!(fold(""), None);
    }
}
