//! Stage oracle: the tokenizer and Whisper's special-token grammar.
//!
//! Skips (rather than fails) when the model isn't in the local cache, so a
//! network-less checkout still runs the rest of the suite. Warm the cache
//! with `ffai models --fetch whisper-tiny-en`.

use ffai_mercury::asr::tokenizer::{WhisperTokenizer, TIMESTAMP_STEP_SECS};

fn tokenizer() -> Option<WhisperTokenizer> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
    let manifests = ffai_models::load_dir(std::path::Path::new(dir)).ok()?;
    let manifest = manifests.into_iter().find(|m| m.name == "whisper-tiny-en")?;
    let path = manifest.local_path("tokenizer.json")?;
    WhisperTokenizer::load(&path).ok()
}

#[test]
fn text_round_trips_through_the_bpe_vocabulary() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: whisper-tiny-en not cached");
        return;
    };
    for text in [
        " Where was the use of imagining?",
        " Mr. Quilter is the apostle of the middle classes.",
        " 1876 was a long time ago.",
    ] {
        let ids = tk.encode(text).expect("encode");
        let back = tk.decode(&ids).expect("decode");
        assert_eq!(back, text, "round-trip changed the text");
    }
}

#[test]
fn control_tokens_are_ordered_and_timestamps_are_contiguous() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: whisper-tiny-en not cached");
        return;
    };
    // Text ids sit below <|endoftext|>; every control token above it.
    assert!(tk.sot > tk.eot, "sot must live in the control range");
    assert!(tk.timestamp_begin > tk.no_timestamps, "timestamps follow the control block");

    // The timestamp grammar: contiguous ids, 20 ms apart, starting at zero.
    assert!(tk.is_timestamp(tk.timestamp_begin));
    assert!(!tk.is_timestamp(tk.timestamp_begin - 1));
    assert_eq!(tk.timestamp_secs(tk.timestamp_begin), 0.0);
    assert!((tk.timestamp_secs(tk.timestamp_begin + 50) - 1.0).abs() < 1e-9);
    assert!((tk.timestamp_secs(tk.timestamp_begin + 1) - TIMESTAMP_STEP_SECS).abs() < 1e-9);
}

#[test]
fn decode_strips_control_tokens_from_text() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: whisper-tiny-en not cached");
        return;
    };
    let mut ids = tk.initial_tokens(None, false, true, true);
    ids.extend(tk.encode(" hello").expect("encode"));
    ids.push(tk.timestamp_begin + 25);
    ids.push(tk.eot);
    assert_eq!(tk.decode(&ids).expect("decode"), " hello");
}

#[test]
fn english_only_prompt_is_bare_sot() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: whisper-tiny-en not cached");
        return;
    };
    // openai-whisper's get_tokenizer(multilingual=False) sets BOTH language
    // and task to None. An .en model must not be handed <|transcribe|>.
    assert_eq!(tk.initial_tokens(None, false, true, true), vec![tk.sot]);
    assert_eq!(
        tk.initial_tokens(None, false, false, true),
        vec![tk.sot, tk.no_timestamps]
    );
}

#[test]
fn multilingual_prompt_carries_language_and_task() {
    let Some(tk) = tokenizer() else {
        eprintln!("skipping: whisper-tiny-en not cached");
        return;
    };
    let lang = 50259; // <|en|> in the multilingual layout
    assert_eq!(
        tk.initial_tokens(Some(lang), false, true, false),
        vec![tk.sot, lang, tk.transcribe]
    );
    assert_eq!(tk.initial_tokens(None, true, true, false), vec![tk.sot, tk.translate]);
}
