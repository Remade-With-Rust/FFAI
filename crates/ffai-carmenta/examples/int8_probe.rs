//! #3 measured experiment: would int8 pay on the CRNN's LSTM shapes?
//! Per LSTM step: two matvecs (1,256)x(256,1024) -- the batch-1
//! decoder shape where Mercury's int8 won 3x. Verdict by measurement,
//! not faith: time f32 matmul vs QMatMul(Q8_0) at the exact shapes.
use candle_core::{Device, Module, Tensor};
use candle_core::quantized::{QMatMul, QTensor, GgmlDType};
use std::time::Instant;

fn main() {
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 1.0, (1024, 256), &dev).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, 256), &dev).unwrap();
    const N: usize = 4000;

    let wt = w.t().unwrap().contiguous().unwrap();
    let t0 = Instant::now();
    for _ in 0..N { let _ = x.matmul(&wt).unwrap(); }
    let f32_us = t0.elapsed().as_micros() as f64 / N as f64;

    let qw = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
    let qm = QMatMul::from_qtensor(qw).unwrap();
    let t0 = Instant::now();
    for _ in 0..N { let _ = Module::forward(&qm, &x).unwrap(); }
    let q8_us = t0.elapsed().as_micros() as f64 / N as f64;

    println!("LSTM gate matvec (1,256)x(256,1024): f32 {f32_us:.1} us, q8_0 {q8_us:.1} us, ratio {:.2}x", f32_us / q8_us);
    println!("verdict: {}", if f32_us / q8_us > 1.3 { "int8 LSTM port is worth a brick" } else { "int8 does not pay at this shape; close the experiment" });
}
