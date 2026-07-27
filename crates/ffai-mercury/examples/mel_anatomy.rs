//! Mel front end, cracked open. 3.67x slower than whisper.cpp's and never
//! examined. Times each stage of MelSpectrogram::compute at the real shape
//! (one 30 s window: 480000 samples -> 3000 frames x 80 mels).
use std::time::Instant;
use ffai_mercury::asr::mel::{MelSpectrogram, N_SAMPLES, N_FFT, HOP_LENGTH};
use rustfft::{num_complex::Complex32, FftPlanner};

fn best<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    std::hint::black_box(f());
    let mut b = f64::MAX;
    for _ in 0..n { let t = Instant::now(); std::hint::black_box(f()); b = b.min(t.elapsed().as_secs_f64()); }
    b
}

fn main() {
    let samples: Vec<f32> = (0..N_SAMPLES)
        .map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
    let front = MelSpectrogram::new(80);

    let whole = best(5, || front.compute(&samples));
    println!("WHOLE mel (30 s window): {:.2} ms   [whisper.cpp: ~4.4 ms/clip]", whole * 1e3);

    let n_frames = N_SAMPLES / HOP_LENGTH;
    let n_bins = N_FFT / 2 + 1;

    // 1. the FFTs alone
    let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut scratch = vec![Complex32::new(0.0, 0.0); N_FFT];
    let t_fft = best(5, || {
        for _ in 0..n_frames { fft.process(&mut scratch); }
    });
    println!("  1. {n_frames} FFTs of size {N_FFT}      {:7.2} ms  ({:.0}%)", t_fft*1e3, t_fft/whole*100.0);

    // 2. the filterbank projection, as our code does it (scalar triple loop)
    let power = vec![1.0f32; n_frames * n_bins];
    let filt = vec![0.5f32; 80 * n_bins];
    let t_fb = best(5, || {
        let mut mel = vec![0f32; 80 * n_frames];
        for m in 0..80 {
            let f = &filt[m*n_bins..(m+1)*n_bins];
            for fr in 0..n_frames {
                let s = &power[fr*n_bins..(fr+1)*n_bins];
                mel[m*n_frames+fr] = f.iter().zip(s).map(|(&w,&p)| w*p).sum();
            }
        }
        mel
    });
    println!("  2. filterbank 80x{n_bins} @ {n_bins}x{n_frames}  {:7.2} ms  ({:.0}%)  {:.1} GFLOP/s",
        t_fb*1e3, t_fb/whole*100.0, 2.0*(80*n_bins*n_frames) as f64/t_fb/1e9);

    // 3. the log10 pass  <- transcendental, the GELU pattern
    let mut mel = vec![1.0f32; 80 * n_frames];
    let t_log = best(20, || {
        let mut peak = f32::MIN;
        for v in mel.iter_mut() { *v = v.max(1e-10).log10(); peak = peak.max(*v); }
        peak
    });
    println!("  3. log10 over {} elems       {:7.2} ms  ({:.0}%)", 80*n_frames, t_log*1e3, t_log/whole*100.0);

    println!("\n  accounted: {:.0}%", (t_fft+t_fb+t_log)/whole*100.0);
}
