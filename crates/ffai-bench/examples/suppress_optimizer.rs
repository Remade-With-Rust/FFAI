//! Suppress-rule optimizer: evaluate drop rules against `net_gain_if_dropped`.
//!
//! The rule-search tool from the Great Gate campaign (§8.106-§8.117). The
//! suppression filter in `ffai-carmenta::suppress` decides which recognised
//! lines the document-text task does not ask for; this is what the candidate
//! rules were tried in before any of them reached Rust. It reads the CSVs
//! harvested by the campaign probes and scores each rule the only way that
//! matters.
//!
//! ## Why the objective is `net_gain_if_dropped` and not accuracy
//!
//! The metric is CER, not classification accuracy, and the two disagree.
//! Dropping an annotated line costs deletions; keeping an orphan costs
//! insertions. A rule with 90 % precision still LOSES if the 10 % it drops is
//! dense body text. `net_gain_if_dropped` is signed per line — `+chars` for an
//! orphan, `-chars` for an annotated line — so summing it over the lines a rule
//! fires on IS the rule's exact character-level effect. Three earlier rules were
//! fitted on accuracy and lost; §8.112's isolated-fragment branch was the first
//! fitted this way and the first to generalise (-0.26 pp train, -0.27 pp
//! holdout).
//!
//! ## Two traps this tool cannot see, which cost the campaign real time
//!
//! Both are properties of the INPUT, not of the rules, so a clean table here can
//! still be void:
//!
//! 1. **Check what the CSV has already been filtered by.** `suppress_wide.csv`
//!    is `w_p90 > 0.198 AND y_rel > 0.0533` — an export taken AFTER the width
//!    branch. A sweep of the width branch on it moves nothing, and the tell is
//!    a constant that is inert across its whole grid (§8.117). The unfiltered
//!    dump is `suppress_lines.csv`.
//! 2. **Sum per PAGE before believing a total.** `year_paren` scored +4 407
//!    characters and was refuted: three pages of 236 carried all of it, and
//!    removing them took the win to -0.01 pp (§8.114). Half the residual in this
//!    corpus sits on ~17 pages of 316 (§8.115), so a large net gain is as likely
//!    to be a page list as a rule. What shipped was the DENSITY-GATED form —
//!    fire only where the page carries >= 4 citation-years — which lifted
//!    precision to 98 % and removed every observed miss (§8.116).
//!
//! A rule that survives both, and agrees in SIGN across train and holdout, has
//! earned a Rust branch. Nothing else has.
//!
//! Usage:
//!   cargo run -p ffai-bench --release --example suppress_optimizer -- \
//!       --input .tools-bench/suppress_wide.csv
//!   cargo run -p ffai-bench --release --example suppress_optimizer -- \
//!       --input .tools-bench/suppress_longprose.csv

use clap::Parser;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "suppress_optimizer")]
#[command(about = "Evaluate suppress rules on net_gain_if_dropped")]
struct Args {
    /// Path to input CSV
    #[arg(short, long)]
    input: PathBuf,
}

/// One harvested line. Fields the current rule sets do not read are kept
/// because they are what the NEXT rule set is built from — the harvest is
/// expensive and the columns are already in the CSVs.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct Row {
    orphan: bool,
    net_gain: f64,

    // Geometry / run features (wide set)
    run_lines: f64,
    w_p90: f64,
    same_left: f64,
    same_left_frac: f64,
    nchars: f64,
    words: f64,
    digit: f64, // fraction 0-1  (wide)
    alpha: f64,

    // Long-prose / bibliographic features
    year_paren: bool,
    et_al: bool,
    lead_num: bool,
    initials: bool,
    journal_abbr: bool,
    doi_url: bool,
    page_range: bool,
    org_word: bool,
    ends_hyphen: bool,
    semicolon_pct: f64,
    comma_pct: f64,
    period_pct: f64,
    digit_pct: f64, // 0-100 style (longprose)
    caps_word_frac: f64,
    mean_word_len: f64,
    conf: f64,
}

#[derive(Debug)]
struct RuleResult {
    name: String,
    n: usize,
    net_gain: f64,
    precision: f64,
    recall: f64,
    true_gain: f64,
    false_cost: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let rows = load_csv(&args.input)?;

    if rows.is_empty() {
        eprintln!("No data rows found.");
        return Ok(());
    }

    let total_orphans: usize = rows.iter().filter(|r| r.orphan).count();
    let perfect_gain: f64 = rows.iter().filter(|r| r.orphan).map(|r| r.net_gain).sum();

    println!(
        "Loaded {} rows  |  orphans: {} ({:.1}%)  |  perfect orphan gain: {:+.0}",
        rows.len(),
        total_orphans,
        100.0 * total_orphans as f64 / rows.len() as f64,
        perfect_gain
    );
    println!();

    // Detect which feature family is present
    let has_longprose = rows.iter().any(|r| r.year_paren || r.caps_word_frac > 0.0);
    let has_wide = rows.iter().any(|r| r.w_p90 > 0.0 || r.run_lines > 0.0);

    if has_wide {
        println!("=== Wide / geometry rules ===");
        run_wide_rules(&rows, total_orphans);
        println!();
    }

    if has_longprose {
        println!("=== Long-prose / bibliographic rules ===");
        run_longprose_rules(&rows, total_orphans);
        println!();
    }

    // Scoring systems
    if has_wide {
        println!("=== Wide scoring system ===");
        run_wide_scoring(&rows, total_orphans);
        println!();
    }
    if has_longprose {
        println!("=== Long-prose scoring system ===");
        run_longprose_scoring(&rows, total_orphans);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Rule sets
// ---------------------------------------------------------------------------

fn run_wide_rules(rows: &[Row], total_orphans: usize) {
    type Pred = Box<dyn Fn(&Row) -> bool>;
    let rules: Vec<(&str, Pred)> = vec![
        ("run_lines <= 2", Box::new(|r| r.run_lines <= 2.0)),
        ("run_lines <= 3", Box::new(|r| r.run_lines <= 3.0)),
        ("w_p90 < 0.35", Box::new(|r| r.w_p90 < 0.35)),
        ("same_left < 5", Box::new(|r| r.same_left < 5.0)),
        (
            "A: run<=3 & w_p90<0.35 & same_left<5",
            Box::new(|r| r.run_lines <= 3.0 && r.w_p90 < 0.35 && r.same_left < 5.0),
        ),
        (
            "B: run<=2 & w_p90<0.30 & same_left<5",
            Box::new(|r| r.run_lines <= 2.0 && r.w_p90 < 0.30 && r.same_left < 5.0),
        ),
        (
            "C: run<=3 & w_p90<0.30 & same_left<5",
            Box::new(|r| r.run_lines <= 3.0 && r.w_p90 < 0.30 && r.same_left < 5.0),
        ),
        (
            "D: run<=3 & w_p90<0.40 & same_left<5 & digit>0.05",
            Box::new(|r| {
                r.run_lines <= 3.0 && r.w_p90 < 0.40 && r.same_left < 5.0 && r.digit > 0.05
            }),
        ),
        (
            "E: run<=2 & w_p90<0.35 & same_left<8 & alpha<0.85",
            Box::new(|r| {
                r.run_lines <= 2.0 && r.w_p90 < 0.35 && r.same_left < 8.0 && r.alpha < 0.85
            }),
        ),
        (
            "w_p90<0.30 & same_left<5",
            Box::new(|r| r.w_p90 < 0.30 && r.same_left < 5.0),
        ),
        (
            "w_p90<0.35 & same_left<5",
            Box::new(|r| r.w_p90 < 0.35 && r.same_left < 5.0),
        ),
        (
            "w_p90<0.25 & same_left<5",
            Box::new(|r| r.w_p90 < 0.25 && r.same_left < 5.0),
        ),
    ];

    print_header();
    let mut results = Vec::new();
    for (name, pred) in &rules {
        let res = evaluate(rows, total_orphans, name, pred.as_ref());
        print_result(&res);
        results.push(res);
    }
    print_best(&results);
}

fn run_longprose_rules(rows: &[Row], total_orphans: usize) {
    type Pred = Box<dyn Fn(&Row) -> bool>;
    let rules: Vec<(&str, Pred)> = vec![
        ("year_paren", Box::new(|r| r.year_paren)),
        ("et_al", Box::new(|r| r.et_al)),
        ("initials", Box::new(|r| r.initials)),
        ("lead_num", Box::new(|r| r.lead_num)),
        ("page_range", Box::new(|r| r.page_range)),
        (
            "year_paren | et_al | initials",
            Box::new(|r| r.year_paren || r.et_al || r.initials),
        ),
        ("caps_word_frac > 0.5", Box::new(|r| r.caps_word_frac > 0.5)),
        ("caps_word_frac > 0.6", Box::new(|r| r.caps_word_frac > 0.6)),
        (
            "year_paren OR (initials & caps>0.35)",
            Box::new(|r| r.year_paren || (r.initials && r.caps_word_frac > 0.35)),
        ),
        (
            "year_paren OR (initials & caps>0.25)",
            Box::new(|r| r.year_paren || (r.initials && r.caps_word_frac > 0.25)),
        ),
        (
            "year | et_al | (initials+caps)",
            Box::new(|r| r.year_paren || r.et_al || (r.initials && r.caps_word_frac > 0.30)),
        ),
        (
            "year_paren OR (caps>0.55 + short run)",
            Box::new(|r| r.year_paren || (r.caps_word_frac > 0.55 && r.run_lines <= 4.0)),
        ),
        (
            "year_paren OR (page_range + caps)",
            Box::new(|r| r.year_paren || (r.page_range && r.caps_word_frac > 0.4)),
        ),
        (
            "year_paren OR doi_url",
            Box::new(|r| r.year_paren || r.doi_url),
        ),
    ];

    print_header();
    let mut results = Vec::new();
    for (name, pred) in &rules {
        let res = evaluate(rows, total_orphans, name, pred.as_ref());
        print_result(&res);
        results.push(res);
    }
    print_best(&results);
}

fn run_wide_scoring(rows: &[Row], total_orphans: usize) {
    print_header_short();
    for thresh in [5usize, 6, 7, 8] {
        let name = format!("score >= {}", thresh);
        let pred = |r: &Row| wide_score(r) >= thresh;
        let res = evaluate(rows, total_orphans, &name, &pred);
        print_result(&res);
    }
}

fn run_longprose_scoring(rows: &[Row], total_orphans: usize) {
    print_header_short();
    for thresh in [3usize, 4, 5, 6] {
        let name = format!("score >= {}", thresh);
        let pred = |r: &Row| longprose_score(r) >= thresh;
        let res = evaluate(rows, total_orphans, &name, &pred);
        print_result(&res);
    }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

fn wide_score(r: &Row) -> usize {
    let mut s = 0usize;
    if r.run_lines <= 1.0 {
        s += 3;
    } else if r.run_lines <= 2.0 {
        s += 2;
    } else if r.run_lines <= 3.0 {
        s += 1;
    }

    if r.w_p90 < 0.25 {
        s += 3;
    } else if r.w_p90 < 0.35 {
        s += 2;
    } else if r.w_p90 < 0.45 {
        s += 1;
    }

    if r.same_left < 5.0 {
        s += 2;
    } else if r.same_left < 10.0 {
        s += 1;
    }

    if r.digit > 0.20 {
        s += 1;
    }
    if r.alpha < 0.60 {
        s += 1;
    }
    s
}

fn longprose_score(r: &Row) -> usize {
    let mut s = 0usize;
    if r.year_paren {
        s += 4;
    }
    if r.et_al {
        s += 2;
    }
    if r.initials {
        s += 2;
    }
    if r.page_range {
        s += 1;
    }
    if r.caps_word_frac > 0.6 {
        s += 2;
    } else if r.caps_word_frac > 0.4 {
        s += 1;
    }
    if r.digit_pct > 8.0 {
        s += 1;
    }
    if r.run_lines <= 2.0 {
        s += 1;
    }
    if r.w_p90 < 0.7 {
        s += 1;
    }
    s
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

fn evaluate(
    rows: &[Row],
    total_orphans: usize,
    name: &str,
    pred: &dyn Fn(&Row) -> bool,
) -> RuleResult {
    let mut n = 0usize;
    let mut net = 0.0f64;
    let mut orphan_hits = 0usize;
    let mut true_gain = 0.0f64;
    let mut false_cost = 0.0f64;

    for r in rows {
        if pred(r) {
            n += 1;
            net += r.net_gain;
            if r.orphan {
                orphan_hits += 1;
                true_gain += r.net_gain;
            } else {
                false_cost += r.net_gain;
            }
        }
    }

    let precision = if n > 0 {
        orphan_hits as f64 / n as f64
    } else {
        0.0
    };
    let recall = if total_orphans > 0 {
        orphan_hits as f64 / total_orphans as f64
    } else {
        0.0
    };

    RuleResult {
        name: name.to_string(),
        n,
        net_gain: net,
        precision,
        recall,
        true_gain,
        false_cost,
    }
}

fn print_header() {
    println!(
        "{:<60} {:>5} {:>9} {:>7} {:>7} {:>8} {:>9}",
        "Rule", "n", "net_gain", "prec", "rec", "trueG", "falseC"
    );
    println!("{}", "-".repeat(110));
}

fn print_header_short() {
    println!(
        "{:<60} {:>5} {:>9} {:>7} {:>7}",
        "Rule", "n", "net_gain", "prec", "rec"
    );
    println!("{}", "-".repeat(95));
}

fn print_result(r: &RuleResult) {
    println!(
        "{:<60} {:>5} {:>+9.0} {:>7.3} {:>7.3} {:>+8.0} {:>+9.0}",
        r.name, r.n, r.net_gain, r.precision, r.recall, r.true_gain, r.false_cost
    );
}

fn print_best(results: &[RuleResult]) {
    if let Some(best) = results
        .iter()
        .filter(|r| r.net_gain > 0.0)
        .max_by(|a, b| a.net_gain.partial_cmp(&b.net_gain).unwrap())
    {
        println!();
        println!(
            "Best positive-gain rule: \"{}\"  ->  net {:+.0}  (prec {:.3}, rec {:.3})",
            best.name, best.net_gain, best.precision, best.recall
        );
        println!(
            "  NOT a result yet: sum it per PAGE and check the sign on BOTH splits \
             first (§8.114, §8.115)."
        );
    }
}

// ---------------------------------------------------------------------------
// CSV loader (schema-agnostic)
// ---------------------------------------------------------------------------

fn load_csv(path: &PathBuf) -> Result<Vec<Row>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(file);

    let headers = rdr.headers()?.clone();
    let mut col: HashMap<String, usize> = HashMap::new();
    for (i, h) in headers.iter().enumerate() {
        col.insert(h.trim().to_lowercase(), i);
    }

    let idx = |name: &str| col.get(name).copied();

    let idx_orphan = idx("orphan").ok_or("Missing column: orphan")?;
    let idx_net = idx("net_gain_if_dropped").ok_or("Missing column: net_gain_if_dropped")?;

    // Optional columns - default to 0 / false if absent
    let idx_run_lines = idx("run_lines");
    let idx_w_p90 = idx("w_p90");
    let idx_same_left = idx("same_left");
    let idx_same_left_frac = idx("same_left_frac");
    let idx_nchars = idx("nchars").or(idx("chars"));
    let idx_words = idx("words");
    let idx_digit = idx("digit");
    let idx_alpha = idx("alpha");

    let idx_year_paren = idx("year_paren");
    let idx_et_al = idx("et_al");
    let idx_lead_num = idx("lead_num");
    let idx_initials = idx("initials");
    let idx_journal_abbr = idx("journal_abbr");
    let idx_doi_url = idx("doi_url");
    let idx_page_range = idx("page_range");
    let idx_org_word = idx("org_word");
    let idx_ends_hyphen = idx("ends_hyphen");
    let idx_semicolon_pct = idx("semicolon_pct");
    let idx_comma_pct = idx("comma_pct");
    let idx_period_pct = idx("period_pct");
    let idx_digit_pct = idx("digit_pct");
    let idx_caps_word_frac = idx("caps_word_frac");
    let idx_mean_word_len = idx("mean_word_len");
    let idx_conf = idx("conf");

    let mut out = Vec::new();
    for result in rdr.records() {
        let record = result?;

        let get_f64 = |opt: Option<usize>| {
            opt.and_then(|i| record.get(i))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0)
        };
        let get_bool = |opt: Option<usize>| {
            opt.and_then(|i| record.get(i))
                .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        };

        let orphan = record
            .get(idx_orphan)
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let net_gain = record
            .get(idx_net)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        out.push(Row {
            orphan,
            net_gain,
            run_lines: get_f64(idx_run_lines),
            w_p90: get_f64(idx_w_p90),
            same_left: get_f64(idx_same_left),
            same_left_frac: get_f64(idx_same_left_frac),
            nchars: get_f64(idx_nchars),
            words: get_f64(idx_words),
            digit: get_f64(idx_digit),
            alpha: get_f64(idx_alpha),
            year_paren: get_bool(idx_year_paren),
            et_al: get_bool(idx_et_al),
            lead_num: get_bool(idx_lead_num),
            initials: get_bool(idx_initials),
            journal_abbr: get_bool(idx_journal_abbr),
            doi_url: get_bool(idx_doi_url),
            page_range: get_bool(idx_page_range),
            org_word: get_bool(idx_org_word),
            ends_hyphen: get_bool(idx_ends_hyphen),
            semicolon_pct: get_f64(idx_semicolon_pct),
            comma_pct: get_f64(idx_comma_pct),
            period_pct: get_f64(idx_period_pct),
            digit_pct: get_f64(idx_digit_pct),
            caps_word_frac: get_f64(idx_caps_word_frac),
            mean_word_len: get_f64(idx_mean_word_len),
            conf: get_f64(idx_conf),
        });
    }
    Ok(out)
}
