//! NT-form scores: read K as (seq, HD) natural layout, no transpose.
//! Current form reads K as (HD, seq) and AXPYs over keys -- needs a transpose.
//! NT needs dot products + horizontal reduction, the structure that measured
//! 46 GFLOP/s vs 73 early on. But with 4x4 register tiling the reduction is
//! amortised 64:1, which the naive version was not.
use std::time::Instant;
use std::arch::x86_64::*;
const HD: usize = 64;
fn t(n: usize, mut f: impl FnMut()) -> f64 {
    f(); let mut b = f64::MAX;
    for _ in 0..n { let s = Instant::now(); f(); b = b.min(s.elapsed().as_secs_f64()); }
    b
}
/// current: kt is (HD, seq), AXPY over keys
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_form(q: &[f32], kt: &[f32], s: &mut [f32], rows: usize, cols: usize, seq: usize) {
    for i in 0..rows { s[i*cols..(i+1)*cols].fill(0.0); }
    for tt in 0..HD {
        let kr = kt.as_ptr().add(tt*seq);
        for i in 0..rows {
            let qv = _mm256_set1_ps(*q.get_unchecked(i*HD+tt));
            let sp = s.as_mut_ptr().add(i*cols);
            let mut j = 0;
            while j+8 <= cols { _mm256_storeu_ps(sp.add(j), _mm256_fmadd_ps(qv, _mm256_loadu_ps(kr.add(j)), _mm256_loadu_ps(sp.add(j)))); j += 8; }
        }
    }
}
/// NT: k is (seq, HD) natural. 4x4 tile, vectorise over HD, 16 reduces per tile.
#[target_feature(enable = "avx2,fma")]
unsafe fn nt_form(q: &[f32], k: &[f32], s: &mut [f32], rows: usize, cols: usize) {
    let mut i = 0;
    while i + 4 <= rows {
        let mut j = 0;
        while j + 4 <= cols {
            let mut acc = [_mm256_setzero_ps(); 16];
            let mut d = 0;
            while d < HD {
                let qv: [__m256; 4] = [
                    _mm256_loadu_ps(q.as_ptr().add((i)*HD+d)),
                    _mm256_loadu_ps(q.as_ptr().add((i+1)*HD+d)),
                    _mm256_loadu_ps(q.as_ptr().add((i+2)*HD+d)),
                    _mm256_loadu_ps(q.as_ptr().add((i+3)*HD+d))];
                for (jj, kvv) in (0..4).map(|jj| (jj, _mm256_loadu_ps(k.as_ptr().add((j+jj)*HD+d)))) {
                    for ii in 0..4 { acc[ii*4+jj] = _mm256_fmadd_ps(qv[ii], kvv, acc[ii*4+jj]); }
                }
                d += 8;
            }
            for ii in 0..4 { for jj in 0..4 {
                let v = acc[ii*4+jj];
                let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v,1));
                let h = _mm_add_ps(h, _mm_movehl_ps(h,h));
                let h = _mm_add_ss(h, _mm_shuffle_ps(h,h,1));
                *s.get_unchecked_mut((i+ii)*cols + j+jj) = _mm_cvtss_f32(h);
            }}
            j += 4;
        }
        i += 4;
    }
}
fn main() {
    let (rows, cols, seq) = (64usize, 256usize, 1500usize);
    let q = vec![0.5f32; rows*HD];
    let kt = vec![0.5f32; HD*seq];
    let k = vec![0.5f32; seq*HD];
    let mut s = vec![0f32; rows*cols];
    let a = t(2000, || unsafe { axpy_form(&q,&kt,&mut s,rows,cols,seq) });
    let b = t(2000, || unsafe { nt_form(&q,&k,&mut s,rows,cols) });
    let f = 2.0*(rows*cols*HD) as f64;
    println!("scores tile {rows}x{cols}, contraction {HD}");
    println!("  AXPY form (needs K transposed) {:7.1} us  {:5.0} GFLOP/s", a*1e6, f/a/1e9);
    println!("  NT form   (K natural, no xpose){:7.1} us  {:5.0} GFLOP/s", b*1e6, f/b/1e9);
    println!("  NT is {:.2}x the AXPY form", a/b);
    println!("\n  prize if NT wins: the K transpose, ~1.27 ms/layer = ~5 ms = 1.7% of pipeline");
    println!("  break-even: NT must be within ~10% of AXPY (kernel is 12.5 ms/layer)");
}
