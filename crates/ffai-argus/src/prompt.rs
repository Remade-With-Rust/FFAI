//! Sequence assembly: turning an image + a question into the exact token
//! sequence the model was trained on.
//!
//! Step 4 of `docs/plans/argus-launch-plan.md`. §3.3 names this
//!
//! > "the actual 'multimodal' step and the one most likely to be silently
//! > wrong (wrong offset = plausible but degraded output)"
//!
//! and §2.2 calls the chat template "the highest-risk silent failure in the
//! whole build". That is not a guess — §7 measured it on this model: **43 of
//! 50 answers changed on identical weights**, from prompt formatting alone,
//! with nothing raising an error.
//!
//! So this module is gated on TOKEN IDS, not on a tolerance. Ids are integers:
//! they match the reference exactly or they do not.
//!
//! # The layout, read off the reference rather than guessed
//!
//! `corpora/refs/dump_smolvlm_prompt.py` dumps what the real processor
//! produces. For a 512x512 image at `scale_factor` 4 that is 1142 tokens:
//!
//! ```text
//! <|im_start|>User:
//!   <fake_token_around_image><row_1_col_1>[64 x <image>]
//!   <fake_token_around_image><row_1_col_2>[64 x <image>]
//!   ...                                                     16 tiles, ROW-MAJOR
//!   <fake_token_around_image><row_4_col_4>[64 x <image>]
//!   "\n\n"
//!   <fake_token_around_image><global-img>[64 x <image>]      the thumbnail is LAST
//!   <fake_token_around_image>
//!   {question}<end_of_utterance>\nAssistant:
//! ```
//!
//! Two details that a reasonable person would get wrong by guessing, and which
//! would produce fluent-but-degraded output rather than an error:
//!
//! * **The global thumbnail comes LAST**, after all 16 tiles — not first, as
//!   "a thumbnail then the detail" would suggest.
//! * **There is a bare `\n\n` between the tile grid and the thumbnail**, and a
//!   closing `<fake_token_around_image>` after the thumbnail before the text.
//!   The `<fake_token_around_image>` count is the check: 16 + 1 + 1 = **18**,
//!   which is what the reference dump reports.

/// The pieces of the layout that vary by checkpoint, read from `config.json`
/// and the processor rather than hard-coded.
#[derive(Debug, Clone)]
pub struct PromptLayout {
    /// Repeated `tokens_per_tile` times per image block.
    pub image_token: String,
    /// Wraps every image block and closes the run.
    pub fake_token: String,
    /// Marks the global thumbnail's block.
    pub global_token: String,
    /// `<row_{r}_col_{c}>`, 1-based.
    pub row_col_fmt: fn(usize, usize) -> String,
    /// `(image_size / patch_size)^2 / scale_factor^2` — 64 for SmolVLM-256M.
    pub tokens_per_tile: usize,
}

impl Default for PromptLayout {
    fn default() -> Self {
        Self {
            image_token: "<image>".into(),
            fake_token: "<fake_token_around_image>".into(),
            global_token: "<global-img>".into(),
            row_col_fmt: |r, c| format!("<row_{r}_col_{c}>"),
            tokens_per_tile: 64,
        }
    }
}

impl PromptLayout {
    /// Derive `tokens_per_tile` from the vision geometry.
    ///
    /// Computed rather than constant: it is
    /// `(image_size / patch_size)^2 / scale_factor^2`, so a different `SmolVLM`
    /// size changes it and a hard-coded 64 would be silently wrong there — the
    /// same class of defect this whole module is guarding against.
    #[must_use]
    pub const fn with_geometry(
        mut self,
        image_size: usize,
        patch_size: usize,
        scale_factor: usize,
    ) -> Self {
        let side = image_size / patch_size;
        self.tokens_per_tile = (side * side) / (scale_factor * scale_factor);
        self
    }

    /// The image-block run: 16 tiles row-major, then `\n\n`, then the global
    /// thumbnail, then a closing fake token.
    ///
    /// `rows`/`cols` describe the tile grid the preprocessor chose. `rows == 0`
    /// means the image was small enough that only the thumbnail exists — the
    /// reference emits just the global block then, with no grid and no `\n\n`.
    #[must_use]
    pub fn image_block(&self, rows: usize, cols: usize) -> String {
        let imgs = self.image_token.repeat(self.tokens_per_tile);
        let mut s = String::new();
        for r in 1..=rows {
            for c in 1..=cols {
                s.push_str(&self.fake_token);
                s.push_str(&(self.row_col_fmt)(r, c));
                s.push_str(&imgs);
            }
            // EVERY row of the grid is newline-terminated, including the last.
            //
            // Found by the token gate rather than by reading. The first
            // assembly omitted these and produced 1139 tokens against the
            // reference's 1142 — three missing terminators, one each for rows
            // 1..3. The fourth was invisible to a structural read because it
            // MERGES with the separator below into a single `\n\n` token, so
            // inspecting the reference's token stream showed one `ĊĊ` and
            // suggested one separator where there are in fact two newlines.
            s.push('\n');
        }
        if rows > 0 {
            // One more before the thumbnail. Adjacent to the final row's
            // terminator this becomes `\n\n`, which the tokenizer emits as the
            // single token the earlier read saw.
            s.push('\n');
        }
        s.push_str(&self.fake_token);
        s.push_str(&self.global_token);
        s.push_str(&imgs);
        s.push_str(&self.fake_token);
        s
    }

    /// The full user turn, chat template included.
    ///
    /// The template is `SmolVLM`'s own — `<|im_start|>User:` … `Assistant:` — and
    /// it is written here only because the tokenizer's Jinja template is not
    /// available to this crate. It is checked against the reference's own
    /// output, which is the only thing that makes writing it acceptable at all.
    #[must_use]
    pub fn user_turn(&self, question: &str, rows: usize, cols: usize) -> String {
        format!(
            "<|im_start|>User:{}{question}<end_of_utterance>\nAssistant:",
            self.image_block(rows, cols)
        )
    }
}

/// How many image tokens a prompt should contain for a given grid.
///
/// The arithmetic the assembly must satisfy, kept separate so a test can state
/// it independently of the string building: every tile plus the global
/// thumbnail contributes `tokens_per_tile`.
#[must_use]
pub const fn expected_image_tokens(layout: &PromptLayout, rows: usize, cols: usize) -> usize {
    (rows * cols + 1) * layout.tokens_per_tile
}

/// How many `<fake_token_around_image>` a prompt should contain.
///
/// One before each tile, one before the thumbnail, one closing the run. The
/// count is a cheap structural check that catches a dropped separator, which a
/// token-count check alone would not.
#[must_use]
pub const fn expected_fake_tokens(rows: usize, cols: usize) -> usize {
    rows * cols + 2
}

/// Splice image embeddings into the text embedding sequence.
///
/// This is §3.3's "actual multimodal step". The reference implements it as a
/// `masked_scatter`: every position where `input_ids == image_token_id` takes
/// the next vector from the image hidden states, in order.
///
/// # Why it is written as an explicit walk rather than a clever gather
///
/// The failure this guards against is an OFF-BY-ONE, and an off-by-one here
/// does not crash — it shifts every image block by one position and yields
/// fluent, plausible, degraded output. So the walk is deliberately literal,
/// and it **fails loudly on any count mismatch** instead of truncating to the
/// shorter of the two, which is exactly how a silent misalignment would enter.
///
/// `image_hidden` is `(tiles, tokens_per_tile, dim)` or any shape whose
/// flattened row count equals the number of image positions; it is consumed in
/// row-major order, matching `masked_scatter` on a contiguous tensor.
///
/// # Errors
/// If the number of image positions differs from the number of supplied image
/// vectors, or if the dimensions disagree.
pub fn merge_image_embeddings(
    text_embeds: &candle_core::Tensor,
    image_hidden: &candle_core::Tensor,
    input_ids: &[i64],
    image_token_id: i64,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core::{IndexOp, Tensor};

    let (batch, seq, dim) = text_embeds.dims3()?;
    if batch != 1 {
        candle_core::bail!("merge expects batch 1, got {batch}");
    }
    if seq != input_ids.len() {
        candle_core::bail!(
            "input_ids has {} tokens but text_embeds has {seq} positions",
            input_ids.len()
        );
    }
    // Flatten the image side to (n_vectors, dim) so tiles are consumed in
    // order, exactly as a contiguous masked_scatter would.
    let img = image_hidden.flatten_to(image_hidden.rank() - 2)?;
    let (n_img, img_dim) = img.dims2()?;
    if img_dim != dim {
        candle_core::bail!("image vectors are {img_dim}-dim but text embeds are {dim}-dim");
    }
    let positions: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == image_token_id)
        .map(|(i, _)| i)
        .collect();
    if positions.len() != n_img {
        candle_core::bail!(
            "{} image positions in the prompt but {n_img} image vectors supplied — \
             a mismatch here would misalign every block that follows, so it is an \
             error rather than a truncation",
            positions.len()
        );
    }

    // Build the merged sequence row by row. `index_select` on the text side
    // plus a scatter would be terser; this is chosen for auditability, and the
    // sequence is ~1k rows so the cost is irrelevant next to a vision tower.
    let text = text_embeds.i(0)?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(seq);
    let mut next = 0usize;
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok == image_token_id {
            rows.push(img.i(next)?);
            next += 1;
        } else {
            rows.push(text.i(i)?);
        }
    }
    debug_assert_eq!(next, n_img, "every image vector must be consumed");
    Tensor::stack(&rows, 0)?.unsqueeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_gives_smolvlm_64_tokens_per_tile() {
        // 512 / 16 = 32 patches a side, 1024 patches, / 4^2 = 64 tokens.
        let l = PromptLayout::default().with_geometry(512, 16, 4);
        assert_eq!(l.tokens_per_tile, 64);
    }

    #[test]
    fn the_block_has_the_counts_the_reference_reports() {
        let l = PromptLayout::default();
        let s = l.image_block(4, 4);
        assert_eq!(
            s.matches("<image>").count(),
            expected_image_tokens(&l, 4, 4),
            "17 blocks of 64"
        );
        assert_eq!(s.matches("<image>").count(), 1088);
        assert_eq!(
            s.matches("<fake_token_around_image>").count(),
            expected_fake_tokens(4, 4),
            "16 tiles + global + closing"
        );
        assert_eq!(s.matches("<fake_token_around_image>").count(), 18);
    }

    /// The ordering detail most likely to be guessed wrong.
    #[test]
    fn the_global_thumbnail_comes_after_every_tile() {
        let s = PromptLayout::default().image_block(4, 4);
        let global = s.find("<global-img>").expect("global marker");
        let last_tile = s.find("<row_4_col_4>").expect("last tile marker");
        assert!(
            global > last_tile,
            "the thumbnail must follow the grid — reversing it produces fluent, \
             degraded output rather than an error"
        );
        // …and there is a bare newline pair between the grid and the thumbnail.
        assert!(s[last_tile..global].contains("\n\n"));
    }

    #[test]
    fn a_thumbnail_only_image_has_no_grid_and_no_separator() {
        let s = PromptLayout::default().image_block(0, 0);
        assert!(!s.contains("<row_"));
        assert!(!s.contains("\n\n"));
        assert_eq!(s.matches("<fake_token_around_image>").count(), 2);
    }

    #[test]
    fn the_user_turn_carries_the_chat_markers() {
        let t = PromptLayout::default().user_turn("What is written in this image?", 4, 4);
        assert!(t.starts_with("<|im_start|>User:"));
        assert!(t.ends_with("<end_of_utterance>\nAssistant:"));
    }
}
