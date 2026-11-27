//! `minerva-server`: Minerva behind an HTTP interface shaped like llama.cpp's `llama-server`, so a
//! client written for it (or for bitnet.cpp's) works unchanged.
//!
//! ```text
//! minerva-server -m MODEL.gguf [--port 8080] [--host 127.0.0.1] [-c CTX] [-t THREADS]
//!                [--lora ADAPTER.gguf [--lora-scale S]] [--head HEADFILE]
//! ```
//!
//! `-ngl`, `-np`, `-tb`, `-fa` and `--no-mmap` are accepted and ignored (CPU only, one slot).
//!
//! Endpoints: `GET /health`, `GET /props`, `POST /completion` (also `/completions`),
//! `POST /tokenize`, `POST /detokenize`, `POST /apply-template`. `/completion` takes `prompt`
//! (text, tokenized with BOS, or an array of ids, used as given), `n_predict`, `temperature`,
//! `top_k`, `top_p`, `min_p`, `seed`, `n_probs`, `stop`, `ignore_eos` and `stream`, with
//! llama-server's defaults. `completion_probabilities` uses llama-server's current shape:
//! per token `{id, token, bytes, logprob, top_logprobs}`, log-probabilities over the full
//! vocabulary. With `stream`, each token is a server-sent event `data: {json}`; the last has
//! `stop: true` and the timings. Requests are served one at a time; the prompt cache keeps the
//! longest prefix shared with the previous request.

#![deny(unsafe_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Instant;

use ffai_minerva::engine::{self, Session};
use ffai_minerva::lora::Adapter;
use ffai_minerva::model::Model;
use ffai_minerva::sampling::{Rng, Sampling, sample, sample_from};
use ffai_minerva::tokenizer::{StreamDecoder, Tokenizer};
use serde_json::{Value, json};

const USAGE: &str = "usage: minerva-server -m MODEL.gguf [--port 8080] [--host 127.0.0.1] [-c CTX] [-t THREADS] [--lora ADAPTER.gguf [--lora-scale S]] [--head HEADFILE]";

struct Args {
    model: String,
    host: String,
    port: u16,
    ctx: usize,
    threads: Option<usize>,
    lora: Option<String>,
    lora_scale: f32,
    head: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        model: String::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        ctx: 4096,
        threads: None,
        lora: None,
        lora_scale: 1.0,
        head: None,
    };
    let mut i = 0;
    let val = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{flag} needs a value"))
    };
    while i < argv.len() {
        let f = argv[i].as_str();
        match f {
            "-m" | "--model" => a.model = val(&mut i, f)?,
            "--host" => a.host = val(&mut i, f)?,
            "--port" => {
                a.port = val(&mut i, f)?
                    .parse()
                    .map_err(|e| format!("--port: {e}"))?
            }
            "-c" | "--ctx-size" => {
                a.ctx = val(&mut i, f)?.parse().map_err(|e| format!("-c: {e}"))?
            }
            "-t" | "--threads" => {
                a.threads = Some(val(&mut i, f)?.parse().map_err(|e| format!("-t: {e}"))?)
            }
            "--lora" => a.lora = Some(val(&mut i, f)?),
            "--lora-scale" => {
                a.lora_scale = val(&mut i, f)?
                    .parse()
                    .map_err(|e| format!("--lora-scale: {e}"))?
            }
            "--head" => a.head = Some(val(&mut i, f)?),
            // llama-server flags with a value that mean nothing here.
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" | "-np" | "--parallel" | "-tb"
            | "--threads-batch" | "-b" | "--batch-size" | "-ub" | "--ubatch-size" | "-fa"
            | "--flash-attn" => {
                let _ = val(&mut i, f)?;
            }
            "--no-mmap" | "--mlock" | "--cont-batching" | "-cb" | "--metrics" | "--no-webui" => {}
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }
    if a.model.is_empty() {
        return Err(format!("-m MODEL.gguf is required\n{USAGE}"));
    }
    if a.ctx < 2 {
        return Err("-c must be at least 2".into());
    }
    Ok(a)
}

/// The model, its tokenizer and the one slot's session, with the prompt cache.
struct Server<'m> {
    m: &'m Model,
    tok: Tokenizer,
    s: Session<'m>,
    /// The tokens whose keys and values the session holds, in order.
    cached: Vec<u32>,
    /// End-of-generation tokens.
    eog: Vec<u32>,
    model_name: String,
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let path = parts
        .next()
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
        {
            len = v
                .trim()
                .parse()
                .map_err(|_| "bad content-length".to_owned())?;
        }
    }
    if len > 64 << 20 {
        return Err("body too large".into());
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, body })
}

fn respond(stream: &mut TcpStream, status: u16, body: &Value) {
    let text = body.to_string();
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{text}",
        text.len()
    );
}

fn error(stream: &mut TcpStream, status: u16, msg: &str) {
    respond(
        stream,
        status,
        &json!({"error": {"code": status, "message": msg}}),
    );
}

/// The prompt as ids: text (tokenized with BOS when the model adds one, specials parsed) or an
/// array of ids and text pieces.
fn prompt_ids(tok: &Tokenizer, p: &Value) -> Result<Vec<u32>, String> {
    match p {
        Value::String(s) => Ok(tok.encode(s, tok.add_bos)),
        Value::Array(items) => {
            let mut out = Vec::new();
            for (i, v) in items.iter().enumerate() {
                match v {
                    Value::Number(n) => out.push(
                        n.as_u64()
                            .and_then(|x| u32::try_from(x).ok())
                            .ok_or("prompt ids must be non-negative integers")?,
                    ),
                    Value::String(s) => out.extend(tok.encode(s, i == 0 && tok.add_bos)),
                    _ => return Err("prompt array items must be ids or text".into()),
                }
            }
            Ok(out)
        }
        _ => Err("prompt must be text or an array of ids".into()),
    }
}

fn token_entry(tok: &Tokenizer, id: u32, lp: f64) -> Value {
    let b = tok.token_bytes(id);
    json!({"id": id, "token": String::from_utf8_lossy(b), "bytes": b, "logprob": lp})
}

/// The length of the end of `text` that could still grow into one of `stops`.
fn partial_stop(text: &str, stops: &[String]) -> usize {
    let mut keep = 0;
    for s in stops {
        for k in (1..s.len()).rev() {
            if k > keep && text.ends_with(&s[..k.min(s.len())]) && s.is_char_boundary(k) {
                keep = k;
                break;
            }
        }
    }
    keep
}

impl Server<'_> {
    fn completion(&mut self, stream: &mut TcpStream, req: &Value) {
        let num = |k: &str, d: f64| req.get(k).and_then(Value::as_f64).unwrap_or(d);
        let ids = match req.get("prompt").map(|p| prompt_ids(&self.tok, p)) {
            Some(Ok(v)) if !v.is_empty() => v,
            Some(Ok(_)) => return error(stream, 400, "the prompt is empty"),
            Some(Err(e)) => return error(stream, 400, &e),
            None => return error(stream, 400, "no prompt"),
        };
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= self.m.cfg.vocab) {
            return error(
                stream,
                400,
                &format!("token {bad} is outside the vocabulary"),
            );
        }
        let ctx = self.s.ctx();
        if ids.len() >= ctx {
            return error(
                stream,
                400,
                &format!(
                    "the prompt ({} tokens) does not fit the context ({ctx})",
                    ids.len()
                ),
            );
        }
        let room = ctx - ids.len();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a token count"
        )]
        let n_predict = match num("n_predict", -1.0) {
            n if n < 0.0 => room,
            n => (n as usize).min(room),
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "small counts"
        )]
        let (top_k, n_probs) = (
            num("top_k", 40.0).max(0.0) as usize,
            num("n_probs", 0.0).max(0.0) as usize,
        );
        #[expect(clippy::cast_possible_truncation, reason = "sampling parameters")]
        let mut sp = Sampling {
            temperature: num("temperature", 0.8) as f32,
            top_k,
            top_p: num("top_p", 0.95) as f32,
            min_p: num("min_p", 0.05) as f32,
            seed: 0,
        };
        let seed = match req.get("seed").and_then(Value::as_i64) {
            Some(s) if s >= 0 => s.cast_unsigned(),
            _ => std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64),
        };
        sp.seed = seed;
        let stops: Vec<String> = match req.get("stop") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect(),
            Some(Value::String(s)) => vec![s.clone()],
            _ => Vec::new(),
        };
        let ignore_eos = req
            .get("ignore_eos")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let streaming = req.get("stream").and_then(Value::as_bool).unwrap_or(false);

        // The prompt cache: keep the longest shared prefix, always re-feeding the last token.
        let shared = self
            .cached
            .iter()
            .zip(&ids)
            .take_while(|(a, b)| a == b)
            .count();
        let keep = shared.min(ids.len() - 1).min(self.s.pos());
        self.s.rewind(keep);
        self.cached.truncate(keep);

        if streaming {
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
            );
        }
        let t0 = Instant::now();
        let mut out_ids = Vec::new();
        let mut probs = Vec::new();
        let mut content = String::new();
        let mut sent = 0usize;
        let mut dec = StreamDecoder::new();
        let mut stop_type = "limit";
        let mut stopping_word = String::new();
        let mut first_ms = 0.0;
        let tok = &self.tok;
        let eog = &self.eog;
        let s = &mut self.s;
        let cached = &mut self.cached;
        let res: Result<(), String> = rayon::scope(|_| {
            s.prefill(&ids[keep..ids.len() - 1])
                .map_err(|e| e.to_string())?;
            cached.extend_from_slice(&ids[keep..ids.len() - 1]);
            let mut rng = Rng::new(seed);
            let mut next = ids[ids.len() - 1];
            for i in 0..n_predict {
                let t = if sp.is_greedy() && n_probs == 0 {
                    s.step_argmax(next).map_err(|e| e.to_string())?
                } else if sp.top_k > 0 && n_probs == 0 {
                    // The head screen's exact top k: the same draw as from the full head.
                    let (cands, _) = s.step_top_k(next, sp.top_k).map_err(|e| e.to_string())?;
                    sample_from(&cands, &sp, &mut rng)
                } else {
                    let logits = s
                        .step(next, true)
                        .map_err(|e| e.to_string())?
                        .unwrap_or_default();
                    let t = sample(logits, &sp, &mut rng);
                    if n_probs > 0 {
                        let norm = engine::log_norm(logits);
                        let top: Vec<Value> = engine::top_k_logprobs(logits, n_probs)
                            .into_iter()
                            .map(|(id, lp)| token_entry(tok, id, lp))
                            .collect();
                        let mut e = token_entry(tok, t, engine::logprob(logits, norm, t));
                        e["top_logprobs"] = Value::Array(top);
                        probs.push(e);
                    }
                    t
                };
                cached.push(next);
                if i == 0 {
                    first_ms = t0.elapsed().as_secs_f64() * 1e3;
                }
                if !ignore_eos && eog.contains(&t) {
                    stop_type = "eos";
                    break;
                }
                out_ids.push(t);
                next = t;
                content.push_str(&dec.push(tok, t));
                if let Some(w) = stops.iter().find(|w| content.contains(w.as_str())) {
                    let at = content.find(w.as_str()).unwrap_or(content.len());
                    content.truncate(at);
                    stop_type = "word";
                    stopping_word.clone_from(w);
                }
                if streaming {
                    let hold = if stop_type == "word" {
                        0
                    } else {
                        partial_stop(&content, &stops)
                    };
                    let upto = content.len() - hold;
                    if upto > sent {
                        let mut ev = json!({"content": &content[sent..upto], "stop": false, "id_slot": 0, "tokens": [t]});
                        if let Some(p) = probs.last() {
                            ev["completion_probabilities"] = json!([p]);
                        }
                        if write!(stream, "data: {ev}\n\n")
                            .and_then(|()| stream.flush())
                            .is_err()
                        {
                            return Ok(()); // The client went away.
                        }
                        sent = upto;
                    }
                }
                if stop_type == "word" {
                    break;
                }
            }
            if stop_type != "word" {
                content.push_str(&dec.finish());
            }
            Ok(())
        });
        if let Err(e) = res {
            return if streaming {
                let _ = write!(stream, "data: {}\n\n", json!({"error": e}));
            } else {
                error(stream, 500, &e)
            };
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1e3;
        let n = out_ids.len() + usize::from(stop_type == "eos");
        let prompt_n = ids.len() - keep;
        #[expect(clippy::cast_precision_loss, reason = "token counts")]
        let timings = json!({
            "prompt_n": prompt_n,
            "prompt_ms": first_ms,
            "prompt_per_token_ms": first_ms / prompt_n.max(1) as f64,
            "prompt_per_second": prompt_n as f64 / (first_ms / 1e3).max(1e-9),
            "predicted_n": n,
            "predicted_ms": total_ms - first_ms,
            "predicted_per_token_ms": (total_ms - first_ms) / n.saturating_sub(1).max(1) as f64,
            "predicted_per_second": n.saturating_sub(1) as f64 / ((total_ms - first_ms) / 1e3).max(1e-9),
        });
        let mut fin = json!({
            "content": if streaming { content.get(sent..).unwrap_or("").to_owned() } else { content.clone() },
            "id_slot": 0,
            "stop": true,
            "model": self.model_name,
            "tokens_predicted": n,
            "tokens_evaluated": ids.len(),
            "tokens_cached": keep,
            "stop_type": stop_type,
            "stopping_word": stopping_word,
            "truncated": false,
            "has_new_line": content.contains('\n'),
            "generation_settings": {
                "n_ctx": ctx, "n_predict": n_predict, "temperature": sp.temperature, "top_k": sp.top_k,
                "top_p": sp.top_p, "min_p": sp.min_p, "seed": seed, "n_probs": n_probs, "ignore_eos": ignore_eos,
            },
            "timings": timings,
        });
        if !streaming {
            fin["tokens"] = json!(out_ids);
            if n_probs > 0 {
                fin["completion_probabilities"] = Value::Array(probs);
            }
            respond(stream, 200, &fin);
        } else {
            let _ = write!(stream, "data: {fin}\n\n");
            let _ = stream.flush();
        }
    }

    fn handle(&mut self, stream: &mut TcpStream) {
        let req = match read_request(stream) {
            Ok(r) => r,
            Err(e) => return error(stream, 400, &e),
        };
        if req.method == "OPTIONS" {
            let _ = write!(
                stream,
                "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            );
            return;
        }
        let body: Value = if req.body.is_empty() {
            json!({})
        } else {
            match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return error(stream, 400, &format!("bad JSON: {e}")),
            }
        };
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/health") => respond(stream, 200, &json!({"status": "ok"})),
            ("GET", "/props") => respond(
                stream,
                200,
                &json!({
                    "default_generation_settings": {"n_ctx": self.s.ctx(), "model": self.model_name},
                    "total_slots": 1,
                    "chat_template": self.tok.chat_template,
                    "model_path": self.model_name,
                }),
            ),
            ("POST", "/completion" | "/completions") => self.completion(stream, &body),
            ("POST", "/tokenize") => {
                let text = body.get("content").and_then(Value::as_str).unwrap_or("");
                let special = body
                    .get("add_special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let ids = self.tok.encode(text, special && self.tok.add_bos);
                if body
                    .get("with_pieces")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    let pieces: Vec<Value> = ids
                        .iter()
                        .map(|&id| json!({"id": id, "piece": String::from_utf8_lossy(self.tok.token_bytes(id))}))
                        .collect();
                    respond(stream, 200, &json!({"tokens": pieces}));
                } else {
                    respond(stream, 200, &json!({"tokens": ids}));
                }
            }
            ("POST", "/detokenize") => {
                let ids: Vec<u32> = body
                    .get("tokens")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().and_then(|x| u32::try_from(x).ok()))
                            .collect()
                    })
                    .unwrap_or_default();
                respond(stream, 200, &json!({"content": self.tok.decode(&ids)}));
            }
            ("POST", "/apply-template") => {
                let msgs: Vec<(String, String)> = body
                    .get("messages")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .map(|m| {
                                let f = |k: &str| {
                                    m.get(k).and_then(Value::as_str).unwrap_or("").to_owned()
                                };
                                (f("role"), f("content"))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                match self.tok.apply_chat_template(&msgs, true) {
                    Ok(p) => respond(stream, 200, &json!({"prompt": p})),
                    Err(e) => error(stream, 400, &e.to_string()),
                }
            }
            _ => error(stream, 404, &format!("no {} {}", req.method, req.path)),
        }
    }
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    if let Some(n) = a.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| e.to_string())?;
    } else if std::env::var_os("RAYON_NUM_THREADS").is_none()
        && let Some(n) = ffai_minerva::threads::recommended()
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| e.to_string())?;
    }
    let listener = TcpListener::bind((a.host.as_str(), a.port))
        .map_err(|e| format!("{}:{}: {e}", a.host, a.port))?;
    let t = Instant::now();
    let m = Model::load(Path::new(&a.model)).map_err(|e| format!("{}: {e}", a.model))?;
    if let Some(h) = &a.head {
        let h8 =
            ffai_minerva::head::HeadQ8::open(&m, Path::new(h)).map_err(|e| format!("{h}: {e}"))?;
        m.screen.attach(h8);
    }
    let adapter = match &a.lora {
        Some(f) => {
            Some(Adapter::load(&m, Path::new(f), a.lora_scale).map_err(|e| format!("{f}: {e}"))?)
        }
        None => None,
    };
    let tok = Tokenizer::from_gguf(m.file_bytes()).map_err(|e| format!("{}: {e}", a.model))?;
    let mut s = Session::new(&m, a.ctx);
    if let Some(ad) = &adapter {
        s = s.with_adapter(ad)?;
    }
    let mut eog: Vec<u32> = tok.eos.into_iter().collect();
    for name in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"] {
        if let Some(id) = tok.special(name)
            && !eog.contains(&id)
        {
            eog.push(id);
        }
    }
    let model_name = Path::new(&a.model)
        .file_name()
        .map_or_else(|| a.model.clone(), |f| f.to_string_lossy().into_owned());
    eprintln!(
        "minerva-server: {model_name} loaded in {:.2}s | ctx {} | threads {} | isa {} | head: {}{} | listening on http://{}:{}",
        t.elapsed().as_secs_f64(),
        a.ctx,
        rayon::current_num_threads(),
        s.isa().name(),
        if a.head.is_some() {
            "sidecar"
        } else {
            "built at first use"
        },
        a.lora
            .as_deref()
            .map_or_else(String::new, |l| format!(" | lora: {l}")),
        a.host,
        a.port
    );
    let mut srv = Server {
        m: &m,
        tok,
        s,
        cached: Vec::new(),
        eog,
        model_name,
    };
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => srv.handle(&mut stream),
            Err(e) => eprintln!("minerva-server: accept: {e}"),
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("minerva-server: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
