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
    /// `net_gain / page_chars` — this line's worth as a fraction of ONE page's
    /// CER. Summed over the lines a rule drops and divided by the page count,
    /// it IS the macro CER delta, in the units the benchmark reports (§8.119).
    /// Zero when the CSV predates the column, in which case macro is not shown.
    macro_gain: f64,
    /// Needed only to count distinct pages, which is macro's denominator.
    page: String,
    /// `train` / `holdout`. A rule is fitted on train and judged once on
    /// holdout; a rule whose splits disagree in SIGN is refused whatever its
    /// total (§8.114 — `year_paren` looked like a -0.61 pp win on holdout and
    /// three pages of 236 carried it).
    split: String,

    // Geometry / run features (wide set)
    run_lines: f64,
    run_chars: f64,
    nn_gap: f64,
    y_rel: f64,
    page_year_hits: f64,
    w_p90: f64,
    same_left: f64,
    same_left_frac: f64,
    nchars: f64,
    words: f64,
    digit: f64, // fraction 0-1  (wide)
    alpha: f64,
    sym: f64,

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
    /// Macro CER delta in percentage points: `100 * sum(macro_gain) / n_pages`.
    macro_pp: f64,
    /// The same figure computed within each split, each over its own page count.
    macro_train: f64,
    macro_hold: f64,
    /// Share of the macro delta carried by its three biggest pages. Approaching
    /// 100 % means the rule is a page list (§8.114, §8.115).
    top3: f64,
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

    if rows.iter().any(|r| r.run_chars > 0.0 && r.y_rel > 0.0) {
        println!("=== CALCULATOR RULE SET, under the full discipline ===");
        run_calculator_rules(&rows, total_orphans);
        println!();
        println!("=== ON TOP OF THE SHIPPED FILTER (the only delta that ships) ===");
        run_incremental(&rows, total_orphans);
        println!();
        println!("=== EXHAUSTIVE CONJUNCTION SEARCH, scored on MACRO ===");
        for depth in [3usize, 4] {
            run_search(&rows, depth);
            println!();
        }
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
// The shipped filter, and incremental rules measured ON TOP of it
// ---------------------------------------------------------------------------

/// `ffai-carmenta::suppress::is_non_body`, transcribed. A candidate's value is
/// what it adds to THIS, not what it scores alone — the standalone tables above
/// double-count everything the shipped branches already drop.
fn shipped(r: &Row) -> bool {
    let by_width = if r.w_p90 <= 0.198 {
        if r.run_chars <= 59.5 { r.nn_gap > 0.0129 } else { r.w_p90 <= 0.0874 }
    } else {
        r.y_rel <= 0.0533
    };
    let isolated_fragment = r.run_lines <= 3.0 && r.w_p90 < 0.35 && r.same_left < 5.0;
    let bibliography = r.year_paren && r.page_year_hits >= 4.0;
    by_width || isolated_fragment || bibliography
}

fn run_incremental(rows: &[Row], total_orphans: usize) {
    type Pred = Box<dyn Fn(&Row) -> bool>;
    // Each entry is the SHIPPED filter OR the candidate, so the delta against
    // the shipped row is what the candidate is actually worth.
    let cands: Vec<(&str, Pred)> = vec![
        ("SHIPPED (baseline)", Box::new(shipped)),
        (
            "+ w_p90<0.30 & same_left<5",
            Box::new(|r| shipped(r) || (r.w_p90 < 0.30 && r.same_left < 5.0)),
        ),
        (
            "+ w_p90<0.25 & same_left<5",
            Box::new(|r| shipped(r) || (r.w_p90 < 0.25 && r.same_left < 5.0)),
        ),
        (
            "+ w_p90<0.35 & same_left<5",
            Box::new(|r| shipped(r) || (r.w_p90 < 0.35 && r.same_left < 5.0)),
        ),
        (
            "+ w_p90<0.30 & same_left<8",
            Box::new(|r| shipped(r) || (r.w_p90 < 0.30 && r.same_left < 8.0)),
        ),
        (
            "+ w_p90<0.30 & same_left<3",
            Box::new(|r| shipped(r) || (r.w_p90 < 0.30 && r.same_left < 3.0)),
        ),
        (
            "+ run_lines<=6 & w_p90<0.30 & same_left<5",
            Box::new(|r| shipped(r) || (r.run_lines <= 6.0 && r.w_p90 < 0.30 && r.same_left < 5.0)),
        ),
    ];
    print_header();
    let mut base = None;
    for (name, pred) in &cands {
        let res = evaluate(rows, total_orphans, name, pred.as_ref());
        print_result(&res);
        if base.is_none() {
            base = Some((res.macro_pp, res.macro_train, res.macro_hold, res.net_gain));
        }
    }
    if let Some((bm, bt, bh, bn)) = base {
        println!("
  DELTA vs shipped (this is the number that decides):");
        for (name, pred) in cands.iter().skip(1) {
            let r = evaluate(rows, total_orphans, name, pred.as_ref());
            let (dm, dt, dh) = (r.macro_pp - bm, r.macro_train - bt, r.macro_hold - bh);
            let verdict = if dt > 0.0 && dh > 0.0 { "BOTH SPLITS" } else { "" };
            println!(
                "    {:<44} {:+7.3} pp macro  ({:+7.3} train, {:+7.3} holdout,                  {:+6.0} chars)  {verdict}",
                name, dm, dt, dh, r.net_gain - bn
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Exhaustive conjunction search (3- and 4-variable)
// ---------------------------------------------------------------------------
//
// Hand-picked rules top out around 2 variables because that is what fits in a
// head. The corpus says the deeper conjunctions are better: the only
// 4-variable rule in the standalone table (run<=3 & w_p90<0.40 & same_left<5 &
// digit>0.05) carries precision 0.880 against 0.77-0.79 for every 2-variable
// rule. Precision is what a conjunction buys -- each added clause removes false
// drops faster than true ones -- and under MACRO a false drop on a
// 200-character slide costs a whole page, so precision is worth more than
// reach.
//
// So enumerate rather than guess. Every predicate is a (feature, threshold,
// direction) triple; a candidate is an AND over 3 or 4 DISTINCT features. Each
// predicate is precomputed as a bitmask over the corpus, so a candidate costs a
// few bitwise ANDs and a walk over the survivors.
//
// Everything is scored INCREMENTALLY -- restricted to the lines the shipped
// filter KEEPS -- because a rule that re-drops what is already dropped is worth
// nothing, and the standalone tables cannot see that.

struct Predicate {
    feature: usize,
    label: String,
    mask: Vec<u64>,
}

fn build_predicates(rows: &[Row]) -> Vec<Predicate> {
    type Get = fn(&Row) -> f64;
    let specs: Vec<(&str, Get, Vec<f64>, bool)> = vec![
        ("w_p90", (|r: &Row| r.w_p90) as Get, vec![0.20, 0.25, 0.30, 0.35, 0.45], true),
        ("same_left", |r: &Row| r.same_left, vec![3.0, 5.0, 8.0, 14.0], true),
        ("run_lines", |r: &Row| r.run_lines, vec![2.0, 3.0, 5.0, 8.0], true),
        ("run_chars", |r: &Row| r.run_chars, vec![60.0, 150.0, 400.0], true),
        ("nchars", |r: &Row| r.nchars, vec![15.0, 30.0, 60.0], true),
        ("words", |r: &Row| r.words, vec![3.0, 6.0, 12.0], true),
        ("conf", |r: &Row| r.conf, vec![0.90, 0.96], true),
        ("mean_word_len", |r: &Row| r.mean_word_len, vec![3.0, 4.0], true),
        ("digit", |r: &Row| r.digit, vec![0.05, 0.15, 0.30], false),
        ("sym", |r: &Row| r.sym, vec![0.10, 0.25], false),
        ("caps_word_frac", |r: &Row| r.caps_word_frac, vec![0.40, 0.60], false),
        ("nn_gap", |r: &Row| r.nn_gap, vec![0.20, 0.60], false),
    ];
    let words = rows.len().div_ceil(64);
    let mut out = Vec::new();
    for (fi, (name, get, thresholds, less)) in specs.iter().enumerate() {
        for &t in thresholds {
            let mut mask = vec![0u64; words];
            for (i, r) in rows.iter().enumerate() {
                let v = get(r);
                if if *less { v < t } else { v > t } {
                    mask[i >> 6] |= 1u64 << (i & 63);
                }
            }
            out.push(Predicate {
                feature: fi,
                label: format!("{name}{}{t}", if *less { "<" } else { ">" }),
                mask,
            });
        }
    }
    out
}

struct Cand {
    label: String,
    n: usize,
    net: f64,
    macro_pp: f64,
    train: f64,
    hold: f64,
    top3: f64,
    prec: f64,
}

fn score_mask(
    rows: &[Row], mask: &[u64], keep: &[u64], n_pages: usize, n_tr: usize, n_ho: usize,
) -> Cand {
    let (mut n, mut net, mut mac, mut mt, mut mh, mut hits) =
        (0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    let mut per: HashMap<&str, f64> = HashMap::new();
    for w in 0..mask.len() {
        let mut bits = mask[w] & keep[w];
        while bits != 0 {
            let i = (w << 6) + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let r = &rows[i];
            n += 1;
            net += r.net_gain;
            mac += r.macro_gain;
            if r.split == "train" {
                mt += r.macro_gain;
            } else {
                mh += r.macro_gain;
            }
            if r.orphan {
                hits += 1;
            }
            *per.entry(r.page.as_str()).or_insert(0.0) += r.macro_gain;
        }
    }
    Cand {
        label: String::new(),
        n,
        net,
        macro_pp: if n_pages > 0 { 100.0 * mac / n_pages as f64 } else { 0.0 },
        train: if n_tr > 0 { 100.0 * mt / n_tr as f64 } else { 0.0 },
        hold: if n_ho > 0 { 100.0 * mh / n_ho as f64 } else { 0.0 },
        top3: top3_share(&per),
        prec: if n > 0 { hits as f64 / n as f64 } else { 0.0 },
    }
}

#[allow(clippy::too_many_arguments)]
fn rec(
    start: usize, slot: usize, depth: usize, combo: &mut Vec<usize>,
    preds: &[Predicate], rows: &[Row], keep: &[u64],
    n_pages: usize, n_tr: usize, n_ho: usize,
    best: &mut Vec<Cand>, tried: &mut usize,
) {
    if slot == depth {
        let mut m = preds[combo[0]].mask.clone();
        for &c in combo.iter().skip(1) {
            for (w, v) in m.iter_mut().enumerate() {
                *v &= preds[c].mask[w];
            }
        }
        *tried += 1;
        let mut c = score_mask(rows, &m, keep, n_pages, n_tr, n_ho);
        // The two gates, applied INSIDE the search rather than after it.
        if c.train > 0.0 && c.hold > 0.0 && c.macro_pp > 0.05 {
            c.label = combo
                .iter()
                .map(|&i| preds[i].label.as_str())
                .collect::<Vec<_>>()
                .join(" & ");
            best.push(c);
        }
        return;
    }
    for p in start..preds.len() {
        if combo[..slot].iter().any(|&c| preds[c].feature == preds[p].feature) {
            continue;
        }
        combo[slot] = p;
        rec(p + 1, slot + 1, depth, combo, preds, rows, keep,
            n_pages, n_tr, n_ho, best, tried);
    }
}

fn run_search(rows: &[Row], depth: usize) {
    let preds = build_predicates(rows);
    let words = rows.len().div_ceil(64);
    // Only the lines the shipped filter KEEPS are in play.
    let mut keep = vec![0u64; words];
    for (i, r) in rows.iter().enumerate() {
        if !shipped(r) {
            keep[i >> 6] |= 1u64 << (i & 63);
        }
    }
    let n_pages = n_pages(rows);
    let (n_tr, n_ho) = split_pages(rows);

    let mut best: Vec<Cand> = Vec::new();
    let mut tried = 0usize;
    let mut combo = vec![0usize; depth];
    rec(0, 0, depth, &mut combo, &preds, rows, &keep,
        n_pages, n_tr, n_ho, &mut best, &mut tried);

    best.sort_by(|a, b| b.macro_pp.partial_cmp(&a.macro_pp).unwrap());
    println!(
        "  {depth}-variable conjunctions: {tried} tried, {} positive on BOTH splits",
        best.len()
    );
    println!(
        "  {:<50} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6} {:>8}",
        "Rule (incremental, on top of shipped)", "n", "MACRO", "train", "holdout",
        "top3", "prec", "chars"
    );
    println!("  {}", "-".repeat(108));
    for c in best.iter().take(12) {
        println!(
            "  {:<50} {:>5} {:>+8.3} {:>+8.3} {:>+8.3} {:>6} {:>6.3} {:>+8.0}",
            c.label, c.n, c.macro_pp, c.train, c.hold, fmt_top3(c.top3), c.prec, c.net
        );
    }
}


/// The rules proposed by the external spreadsheet analysis, scored under the
/// full discipline rather than on net characters alone.
///
/// They reproduce this tool exactly where they overlap -- "Pure geometry A"
/// reads 2599 / +6303 / 0.773 in both, and `year_paren` reads 126 / +5804 /
/// 0.841 -- so the objective is implemented the same way on both sides. What
/// the spreadsheet cannot see is what the rest of these columns say.
fn run_calculator_rules(rows: &[Row], total_orphans: usize) {
    type Pred = Box<dyn Fn(&Row) -> bool>;
    let geo = |r: &Row| r.run_lines <= 3.0 && r.w_p90 < 0.35 && r.same_left < 5.0;
    let rules: Vec<(&str, Pred)> = vec![
        (
            "1 year_paren OR (run<=3 & w<0.35 & sl<5)",
            Box::new(move |r| r.year_paren || geo(r)),
        ),
        (
            "2 year_paren OR (w<0.25 & sl<5)",
            Box::new(|r| r.year_paren || (r.w_p90 < 0.25 && r.same_left < 5.0)),
        ),
        (
            "3 year_paren OR (run<=2 & w<0.30 & sl<5)",
            Box::new(|r| {
                r.year_paren || (r.run_lines <= 2.0 && r.w_p90 < 0.30 && r.same_left < 5.0)
            }),
        ),
        ("4 pure geometry A", Box::new(move |r| geo(r))),
        ("5 year_paren alone", Box::new(|r| r.year_paren)),
        (
            "6 ultra-precise: run<=2 & w<0.25 & sl<5 & digit>0.2",
            Box::new(|r| {
                r.run_lines <= 2.0 && r.w_p90 < 0.25 && r.same_left < 5.0 && r.digit > 0.2
            }),
        ),
        // The same #1, with the citation-year clause DENSITY-GATED as shipped.
        (
            "1b same, but year_paren gated at >=4/page (§8.116)",
            Box::new(move |r| (r.year_paren && r.page_year_hits >= 4.0) || geo(r)),
        ),
    ];
    print_header();
    let mut results = Vec::new();
    for (name, pred) in &rules {
        let res = evaluate(rows, total_orphans, name, pred.as_ref());
        print_result(&res);
        results.push(res);
    }

    // What each adds ON TOP of what already ships. The shipped filter already
    // contains the geometry clause verbatim (the §8.112 isolated-fragment
    // branch) and a density-gated year_paren, so most of these totals are
    // re-counting work already done.
    let base = evaluate(rows, total_orphans, "shipped", &shipped);
    println!("\n  vs the SHIPPED filter (which already contains both clauses):");
    println!(
        "  {:<48} {:>9} {:>9} {:>8} {:>8}",
        "", "macro pp", "vs ship", "train", "holdout"
    );
    println!("  {:<48} {:>+9.3} {:>9} {:>+8.3} {:>+8.3}",
             "SHIPPED", base.macro_pp, "-", base.macro_train, base.macro_hold);
    for r in &results {
        println!(
            "  {:<48} {:>+9.3} {:>+9.3} {:>+8.3} {:>+8.3}",
            r.name, r.macro_pp, r.macro_pp - base.macro_pp, r.macro_train, r.macro_hold
        );
    }
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
    let n_pages = n_pages(rows);
    let (n_tr, n_ho) = split_pages(rows);
    let mut n = 0usize;
    let mut net = 0.0f64;
    let mut mac = 0.0f64;
    let mut mac_tr = 0.0f64;
    let mut mac_ho = 0.0f64;
    let mut per_page: HashMap<&str, f64> = HashMap::new();
    let mut orphan_hits = 0usize;
    let mut true_gain = 0.0f64;
    let mut false_cost = 0.0f64;

    for r in rows {
        if pred(r) {
            n += 1;
            net += r.net_gain;
            mac += r.macro_gain;
            if r.split == "train" {
                mac_tr += r.macro_gain;
            } else {
                mac_ho += r.macro_gain;
            }
            *per_page.entry(r.page.as_str()).or_insert(0.0) += r.macro_gain;
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
        macro_pp: if n_pages > 0 { 100.0 * mac / n_pages as f64 } else { 0.0 },
        macro_train: if n_tr > 0 { 100.0 * mac_tr / n_tr as f64 } else { 0.0 },
        macro_hold: if n_ho > 0 { 100.0 * mac_ho / n_ho as f64 } else { 0.0 },
        top3: top3_share(&per_page),
        precision,
        recall,
        true_gain,
        false_cost,
    }
}

/// Pages per split, each macro figure's own denominator.
fn split_pages(rows: &[Row]) -> (usize, usize) {
    let mut tr = std::collections::HashSet::new();
    let mut ho = std::collections::HashSet::new();
    for r in rows {
        if r.split == "train" { tr.insert(r.page.as_str()); } else { ho.insert(r.page.as_str()); }
    }
    (tr.len(), ho.len())
}

/// Of the macro delta a rule produces, the share carried by its three biggest
/// pages. `year_paren` reads ~100 % here, which is what §8.114 had to discover
/// the expensive way.
fn top3_share(per_page: &HashMap<&str, f64>) -> f64 {
    let tot: f64 = per_page.values().sum();
    let gross: f64 = per_page.values().map(|v| v.abs()).sum();
    // A net that is a near-cancellation of large opposite movements makes this
    // ratio explode (-3207 % was observed). That is not a concentration reading,
    // it is a division by almost zero — report NaN and let the caller print n/a.
    if gross < 1e-12 || tot.abs() < 0.02 * gross {
        return f64::NAN;
    }
    let mut v: Vec<f64> = per_page.values().copied().collect();
    v.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    100.0 * v.iter().take(3).sum::<f64>() / tot
}

/// `top3` as a column: `n/a` when the net is a near-cancellation.
fn fmt_top3(v: f64) -> String {
    if v.is_nan() { "  n/a".into() } else { format!("{v:4.0}%") }
}

/// Macro's denominator: a page counts once, whatever its length.
fn n_pages(rows: &[Row]) -> usize {
    rows.iter().map(|r| r.page.as_str()).collect::<std::collections::HashSet<_>>().len()
}

fn print_header() {
    println!(
        "{:<48} {:>5} {:>9} {:>9} {:>8} {:>8} {:>6} {:>6}",
        "Rule", "n", "net_gain", "MACRO pp", "train", "holdout", "top3", "prec"
    );
    println!("{}", "-".repeat(105));
}

fn print_header_short() {
    println!(
        "{:<48} {:>5} {:>9} {:>9} {:>8} {:>8} {:>6} {:>6}",
        "Rule", "n", "net_gain", "MACRO pp", "train", "holdout", "top3", "prec"
    );
    println!("{}", "-".repeat(105));
}

fn print_result(r: &RuleResult) {
    println!(
        "{:<48} {:>5} {:>+9.0} {:>+9.3} {:>+8.3} {:>+8.3} {:>6} {:>6.3}",
        r.name, r.n, r.net_gain, r.macro_pp, r.macro_train, r.macro_hold,
        fmt_top3(r.top3), r.precision
    );
}

fn print_best(results: &[RuleResult]) {
    // Ranked by MACRO, because that is what the benchmark aggregates
    // (`summary.cer = mean(&cers)`). Ranking by characters optimises MICRO, and
    // §8.119 found the whole campaign fitted to the wrong one. The two do not
    // merely differ in size — they pick different rules and flip signs.
    let best_macro = results
        .iter()
        .filter(|r| r.macro_pp > 0.0)
        .max_by(|a, b| a.macro_pp.partial_cmp(&b.macro_pp).unwrap());
    let best_micro = results
        .iter()
        .filter(|r| r.net_gain > 0.0)
        .max_by(|a, b| a.net_gain.partial_cmp(&b.net_gain).unwrap());

    println!();
    if let Some(b) = best_macro {
        println!(
            "Best by MACRO: \"{}\"  ->  {:+.3} pp   (micro {:+.0} chars, prec {:.3})",
            b.name, b.macro_pp, b.net_gain, b.precision
        );
    }
    if let Some(b) = best_micro {
        println!(
            "Best by micro: \"{}\"  ->  {:+.0} chars   ({:+.3} pp macro, prec {:.3})",
            b.name, b.net_gain, b.macro_pp, b.precision
        );
    }
    if let (Some(a), Some(b)) = (best_macro, best_micro) {
        if a.name != b.name {
            println!("  ^^ THE OBJECTIVES DISAGREE — macro is the one that scores.");
        }
    }

    // Sign disagreement between the objectives is the single most useful thing
    // this table produces: a rule that loses characters while winning pages is
    // invisible to every search this campaign ran before §8.119.
    let flips: Vec<&RuleResult> = results
        .iter()
        .filter(|r| r.net_gain * r.macro_pp < 0.0)
        .collect();
    if !flips.is_empty() {
        println!("\n  SIGN FLIPS between micro and macro ({}):", flips.len());
        for r in flips {
            let dir = if r.macro_pp > 0.0 { "macro WINS" } else { "macro LOSES" };
            println!("    {:<48} {:+8.0} chars  {:+7.3} pp   {dir}", r.name, r.net_gain, r.macro_pp);
        }
    }

    // The two gates §8.114 and §8.115 were bought at cost. Neither is optional.
    let ok: Vec<&RuleResult> = results
        .iter()
        .filter(|r| r.macro_train > 0.0 && r.macro_hold > 0.0)
        .collect();
    println!("\n  Rules positive on BOTH splits (macro): {}", ok.len());
    for r in ok {
        println!(
            "    {:<48} train {:+7.3}  holdout {:+7.3}  top3 {}",
            r.name, r.macro_train, r.macro_hold, fmt_top3(r.top3)
        );
    }
    println!(
        "  Still not a result: a rule whose top3 share approaches 100 % is a page \
         list, not a rule (§8.114)."
    );
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
    let idx_page = idx("page");
    let idx_split = idx("split");
    // Present only in `great_gate_full.csv`; older harvests score micro only.
    let idx_macro = idx("macro_gain");
    let idx_page_chars = idx("page_chars");
    let idx_net = idx("net_gain_if_dropped").ok_or("Missing column: net_gain_if_dropped")?;

    // Optional columns - default to 0 / false if absent
    let idx_run_lines = idx("run_lines");
    let idx_run_chars = idx("run_chars");
    let idx_nn_gap = idx("nn_gap");
    let idx_y_rel = idx("y_rel");
    let idx_page_year_hits = idx("page_year_hits");
    let idx_w_p90 = idx("w_p90");
    let idx_same_left = idx("same_left");
    let idx_same_left_frac = idx("same_left_frac");
    let idx_nchars = idx("nchars").or(idx("chars"));
    let idx_words = idx("words");
    let idx_digit = idx("digit");
    let idx_sym = idx("sym");
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

        // Prefer the exported column; fall back to deriving it, so a CSV that
        // carries `page_chars` but not `macro_gain` still scores macro.
        let macro_gain = match idx_macro {
            Some(i) => record.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            None => {
                let pc: f64 = get_f64(idx_page_chars);
                if pc > 0.0 { net_gain / pc } else { 0.0 }
            }
        };

        out.push(Row {
            orphan,
            net_gain,
            macro_gain,
            page: idx_page.and_then(|i| record.get(i)).unwrap_or("").to_string(),
            split: idx_split.and_then(|i| record.get(i)).unwrap_or("holdout").to_string(),
            run_lines: get_f64(idx_run_lines),
            run_chars: get_f64(idx_run_chars),
            nn_gap: get_f64(idx_nn_gap),
            y_rel: get_f64(idx_y_rel),
            page_year_hits: get_f64(idx_page_year_hits),
            w_p90: get_f64(idx_w_p90),
            same_left: get_f64(idx_same_left),
            same_left_frac: get_f64(idx_same_left_frac),
            nchars: get_f64(idx_nchars),
            words: get_f64(idx_words),
            digit: get_f64(idx_digit),
            sym: get_f64(idx_sym),
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
