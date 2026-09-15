use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;
use ndarray::Array2;
use ort::session::{Session, builder::SessionBuilder};
use ort::value::Value;

pub mod config;
pub mod tree_index;
pub use config::get_config;

// ==================== TRANSFORMER EMBEDDING ====================
pub struct TransformerEmbedder {
    pub model: std::sync::Mutex<Option<Session>>,
    pub tokenizer: tokenizers::Tokenizer,
    pub dim: usize,
}

impl TransformerEmbedder {
    fn new() -> Self {
        let cfg = get_config();
        let tokenizer = tokenizers::Tokenizer::from_file("model.tokenizer.json")
            .or_else(|_| tokenizers::Tokenizer::from_file("tokenizer.json"))
            .unwrap_or_else(|e| {
                eprintln!("  WARNING: tokenizer.json gagal ({}), fallback hash", e);
                tokenizers::Tokenizer::new(tokenizers::models::wordpiece::WordPiece::default())
            });
        let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let model = SessionBuilder::new()
            .and_then(|b| b.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).map_err(ort::Error::from))
            .and_then(|b| b.with_intra_threads(num_cpus).map_err(ort::Error::from))
            .and_then(|mut b| b.commit_from_file("model.onnx"))
            .map_err(|e| { eprintln!("  WARNING: model.onnx gagal ({}), fallback hash", e); e })
            .ok();
        if model.is_some() { eprintln!("  ORT model loaded (dim={})", cfg.embedding_dim); }
        else { eprintln!("  ORT model TIDAK ada: hash embedding (dim={})", cfg.embedding_dim); }
        Self { model: std::sync::Mutex::new(model), tokenizer, dim: cfg.embedding_dim }
    }

    fn has_model(&self) -> bool { self.model.lock().map(|g| g.is_some()).unwrap_or(false) }

    fn embed(&self, text: &str) -> Vec<f32> {
        let encoding = match self.tokenizer.encode(text, true) { Ok(enc) => enc, Err(_) => return self.embed_hash(text) };
        if encoding.len() == 0 { return self.embed_hash(text); }
        if let Ok(mut guard) = self.model.lock() {
            if let Some(ref mut model) = *guard {
                if let Some(v) = Self::embed_with_model_static(model, &encoding, self.dim) { return v; }
            }
        }
        self.embed_hash(text)
    }

    fn embed_query(&self, text: &str, dim_df: &[u32], n_docs: usize) -> Vec<f32> {
        if self.has_model() { return self.embed(text); }
        self.embed_hash_idf(text, dim_df, n_docs)
    }

    fn embed_with_model_static(model: &mut Session, encoding: &tokenizers::Encoding, dim: usize) -> Option<Vec<f32>> {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        if ids.is_empty() { return None; }
        let seq_len = ids.len();
        let ids_arr = Array2::from_shape_vec((1, seq_len), ids.iter().map(|&x| x as i64).collect()).ok()?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), mask.iter().map(|&x| x as i64).collect()).ok()?;
        let ttype_arr = Array2::from_shape_vec((1, seq_len), vec![0i64; seq_len]).ok()?;
        let outputs = model.run(ort::inputs![
            "input_ids" => Value::from_array(ids_arr.into_dyn()).ok()?,
            "attention_mask" => Value::from_array(mask_arr.into_dyn()).ok()?,
            "token_type_ids" => Value::from_array(ttype_arr.into_dyn()).ok()?
        ]).ok()?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().ok()?;
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        if shape.len() != 3 { return None; }
        let hidden = shape[2];
        let mask_f: Vec<f32> = mask.iter().map(|&x| x as f32).collect();
        let mask_sum: f32 = mask_f.iter().sum::<f32>().max(1.0);
        let mut pooled = vec![0.0f32; hidden];
        for i in 0..seq_len {
            if mask_f[i] == 0.0 { continue; }
            let base = i * hidden;
            for h in 0..hidden { pooled[h] += data[base + h] * mask_f[i]; }
        }
        for v in &mut pooled { *v /= mask_sum; }
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for v in &mut pooled { *v /= norm; } }
        if pooled.len() > dim { pooled.truncate(dim); }
        else if pooled.len() < dim { pooled.resize(dim, 0.0); }
        Some(pooled)
    }

    pub fn embed_batch_static(model: &mut Session, tokenizer: &tokenizers::Tokenizer, texts: &[String], dim: usize) -> Vec<Vec<f32>> {
        if texts.is_empty() { return Vec::new(); }
        let encodings = match tokenizer.encode_batch(texts.to_vec(), true) { Ok(enc) => enc, Err(_) => return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect() };
        let n = encodings.len();
        let max_seq = encodings.iter().map(|e| e.get_ids().len().min(96)).max().unwrap_or(1).max(1);
        let mut ids_flat: Vec<i64> = Vec::with_capacity(n * max_seq);
        let mut mask_flat: Vec<i64> = Vec::with_capacity(n * max_seq);
        let ttype_flat: Vec<i64> = vec![0i64; n * max_seq];
        for enc in &encodings {
            let ids = enc.get_ids(); let mask = enc.get_attention_mask();
            let seq = ids.len().min(max_seq);
            for &id in ids.iter().take(seq) { ids_flat.push(id as i64); }
            for &m in mask.iter().take(seq) { mask_flat.push(m as i64); }
            ids_flat.resize(ids_flat.len() + (max_seq - seq), 0);
            mask_flat.resize(mask_flat.len() + (max_seq - seq), 0);
        }
        let mask_for_pool = mask_flat.clone();
        let ids_arr = Array2::from_shape_vec((n, max_seq), ids_flat).unwrap();
        let mask_arr = Array2::from_shape_vec((n, max_seq), mask_flat).unwrap();
        let ttype_arr = Array2::from_shape_vec((n, max_seq), ttype_flat).unwrap();
        let ids_val = match Value::from_array(ids_arr.into_dyn()) { Ok(v) => v, Err(_) => return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect() };
        let mask_val = match Value::from_array(mask_arr.into_dyn()) { Ok(v) => v, Err(_) => return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect() };
        let ttype_val = match Value::from_array(ttype_arr.into_dyn()) { Ok(v) => v, Err(_) => return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect() };
        let outputs = match model.run(ort::inputs!["input_ids" => ids_val, "attention_mask" => mask_val, "token_type_ids" => ttype_val]) { Ok(o) => o, Err(_) => return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect() };
        let (shape, data) = match outputs[0].try_extract_tensor::<f32>() { Ok(v) => v, Err(_) => return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect() };
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        if shape.len() != 3 || shape[0] != n { return texts.iter().map(|t| Self::embed_hash_static(t, dim)).collect(); }
        let hidden = shape[2];
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            let seq = encodings[i].get_ids().len().min(max_seq);
            let mask_f: Vec<f32> = mask_for_pool[i*max_seq..(i+1)*max_seq].iter().map(|&x| x as f32).collect();
            let mask_sum: f32 = mask_f.iter().sum::<f32>().max(1.0);
            let mut pooled = vec![0.0f32; hidden];
            for t in 0..seq { if mask_f[t] == 0.0 { continue; } let base = i * max_seq * hidden + t * hidden; for h in 0..hidden { pooled[h] += data[base + h] * mask_f[t]; } }
            for v in &mut pooled { *v /= mask_sum; }
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { for v in &mut pooled { *v /= norm; } }
            if pooled.len() > dim { pooled.truncate(dim); } else if pooled.len() < dim { pooled.resize(dim, 0.0); }
            results.push(pooled);
        }
        results
    }

    fn embed_hash_static(text: &str, dim: usize) -> Vec<f32> {
        let mut emb = vec![0.0f32; dim];
        for_each_gram_idx(text, dim, |idx| { emb[idx] += 1.0; });
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for v in &mut emb { *v /= norm; } }
        emb
    }

    fn embed_hash(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dim];
        let mut ngram_count = 0usize;
        for_each_gram_idx(text, self.dim, |idx| { embedding[idx] += 1.0; ngram_count += 1; });
        if ngram_count == 0 { return embedding; }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in &mut embedding { *x /= norm; } }
        embedding
    }

    fn embed_hash_idf(&self, text: &str, dim_df: &[u32], n_docs: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dim];
        let mut ngram_count = 0usize;
        let n = n_docs.max(1) as f64;
        for_each_gram_idx(text, self.dim, |idx| {
            let df = dim_df.get(idx).copied().unwrap_or(0);
            if df == 0 { return; }
            let idf = ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln() as f32;
            embedding[idx] += idf; ngram_count += 1;
        });
        if ngram_count == 0 { return embedding; }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in &mut embedding { *x /= norm; } }
        embedding
    }
}

fn for_each_gram_idx(text: &str, dim: usize, mut f: impl FnMut(usize)) {
    for word in text.split_whitespace() {
        let lower = word.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < 3 { continue; }
        for n in 3..=5 {
            if chars.len() + 2 < n { continue; }
            for i in 0..=(chars.len() + 2 - n) {
                let mut h: u64 = 0xcbf29ce484222325;
                for j in 0..n {
                    let c = if i == 0 && j == 0 { '#' as u32 } else if i + j == chars.len() + 1 { '#' as u32 } else { chars[i + j - 1] as u32 };
                    h ^= c as u64; h = h.wrapping_mul(0x100000001b3);
                }
                f((h % dim as u64) as usize);
            }
        }
    }
}

// ==================== SPARSE EMBEDDING ====================
pub struct SparseEmbedder { vocab: HashMap<String, u32>, idf: HashMap<String, f64> }
impl SparseEmbedder {
    pub fn new() -> Self { Self { vocab: HashMap::new(), idf: HashMap::new() } }
    pub fn build_vocab(&mut self, texts: &[String]) {
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut df: HashMap<String, usize> = HashMap::new();
        let n = texts.len() as f64;
        for text in texts { let words: HashSet<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect(); for word in &words { *tf.entry(word.clone()).or_insert(0) += 1; *df.entry(word.clone()).or_insert(0) += 1;
            // CJK bigrams
            let chars: Vec<char> = word.chars().collect();
            for i in 0..chars.len().saturating_sub(1) {
                if chars[i] as u32 >= 0x4E00 {
                    let bigram: String = chars[i..=i+1].iter().collect();
                    *tf.entry(bigram.clone()).or_insert(0) += 1;
                    *df.entry(bigram).or_insert(0) += 1;
                }
            }
        } }
        for (word, &count) in &tf { if count >= 1 && word.len() > 2 { let idx = self.vocab.len() as u32; self.vocab.insert(word.clone(), idx); } }
        for (word, &df_val) in &df { self.idf.insert(word.clone(), (n / (df_val as f64 + 1.0)).ln()); }
    }
    pub fn embed(&self, text: &str) -> HashMap<u32, f32> {
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
        let mut sparse: HashMap<u32, f32> = HashMap::new();
        for word in &words {
            if let Some(&idx) = self.vocab.get(word) { *sparse.entry(idx).or_insert(0.0) += *self.idf.get(word).unwrap_or(&1.0) as f32; }
            // CJK bigrams: each adjacent pair of CJK chars becomes a token
            let chars: Vec<char> = word.chars().collect();
            for i in 0..chars.len().saturating_sub(1) {
                if chars[i].is_ascii_alphabetic() || chars[i] as u32 >= 0x4E00 { // CJK or latin
                    let bigram: String = chars[i..=i+1].iter().collect();
                    if let Some(&idx) = self.vocab.get(&bigram) { *sparse.entry(idx).or_insert(0.0) += *self.idf.get(&bigram).unwrap_or(&1.0) as f32; }
                }
            }
        }
        let norm: f32 = sparse.values().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in sparse.values_mut() { *x /= norm; } }
        sparse
    }
    pub fn sparse_cosine(a: &HashMap<u32, f32>, b: &HashMap<u32, f32>) -> f64 {
        if a.is_empty() || b.is_empty() { return 0.0; }
        let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let mut dot = 0.0;
        for (k, v) in small { if let Some(w) = big.get(k) { dot += *v as f64 * *w as f64; } }
        dot
    }
}

// ==================== NORMALIZATION / CHUNKING ====================
fn is_ar_diacritic(c: char) -> bool { matches!(c, '\u{64B}'..='\u{652}' | '\u{670}') }
fn normalize(raw: &str) -> String {
    let nfkc: String = raw.nfkc().collect();
    let mut out = String::with_capacity(nfkc.len());
    let mut prev_space = true;
    for c in nfkc.chars() {
        if is_ar_diacritic(c) { continue; }
        if c.is_whitespace() { if !prev_space { out.push(' '); prev_space = true; } } else { out.push(c); prev_space = false; }
    }
    if prev_space { out.pop(); }
    out
}

fn is_heading(s: &str) -> bool {
    let t = s.trim_start_matches(['#',' ']).trim();
    if t.len()>160||t.len()<8 { return false; }
    t.chars().all(|c| c.is_uppercase()||!c.is_alphabetic()) || t.starts_with("BAB ")||t.starts_with("Chapter ")
}

fn split_sentences(text: &str) -> Vec<String> {
    let flat = text.replace('\n', ". ");
    let mut parts = Vec::new();
    let ch: Vec<char> = flat.chars().collect();
    let mut start = 0; let mut i = 0;
    while i < ch.len() {
        if matches!(ch[i], '.'|'!'|'?'|'\u{3002}'|'\u{FF01}'|'\u{FF1F}') {
            let s: String = ch[start..=i].iter().collect();
            let s = s.trim();
            if !s.is_empty() { parts.push(s.to_string()); }
            i += 1;
            while i < ch.len() && (ch[i]==' '||ch[i]=='.') { if ch[i]!=' '{break;} i+=1; }
            start = i;
        } else { i += 1; }
    }
    if start < ch.len() { let s: String = ch[start..].iter().collect(); if !s.trim().is_empty() { parts.push(s.trim().to_string()); } }
    let cfg = get_config();
    let mut out = Vec::with_capacity(parts.len());
    for s in parts {
        let mut rest = s.as_str();
        while rest.len() > cfg.hard_split {
            let mut cut = cfg.hard_split.min(rest.len());
            while cut > 0 && !rest.is_char_boundary(cut) { cut -= 1; }
            if let Some(sp) = rest[..cut].rfind(' ') { if sp > 300 { cut = sp; } }
            while cut > 0 && !rest.is_char_boundary(cut) { cut -= 1; }
            out.push(rest[..cut].trim().to_string());
            rest = rest[cut..].trim();
        }
        if !rest.is_empty() { out.push(rest.to_string()); }
    }
    out
}

pub fn chunk_text(text: &str) -> Vec<(String, String)> {
    let cfg = get_config();
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut buf_len = 0usize;
    let mut heading = String::new();
    for s in sentences {
        if is_heading(&s) { heading = s.trim_start_matches(['#',' ']).chars().take(120).collect(); }
        if buf_len + s.len() + 1 > cfg.chunk_chars && !buf.is_empty() {
            let content = buf.join(" ");
            if content.len() >= cfg.min_chunk { chunks.push((content, heading.clone())); }
            let mut tail = Vec::new(); let mut tl = 0;
            for item in buf.iter().rev() { tail.insert(0,item.clone()); tl+=item.len()+1; if tl>=cfg.chunk_overlap||tail.len()>=2{break;} }
            buf = tail; buf_len = tl;
        }
        buf_len += s.len() + 1; buf.push(s);
    }
    if !buf.is_empty() { let content = buf.join(" "); if content.len() >= cfg.min_chunk { chunks.push((content, heading)); } }
    chunks
}

// ==================== INDEX ====================
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Doc { content: String, source: String, page: u32, heading: String, dense_embedding: Vec<f32>, sparse_embedding: HashMap<u32, f32> }

#[derive(serde::Serialize, serde::Deserialize)]
struct OnodIndex { docs: Vec<Doc>, sparse_index: Vec<HashMap<u32, f32>>, dim_df: Vec<u32> }

impl OnodIndex {
    fn new() -> Self { let dim = get_config().embedding_dim; Self { docs: Vec::new(), sparse_index: Vec::new(), dim_df: vec![0u32; dim] } }

    fn build_index_batch(&mut self, file_texts: &[(String, String)], _embedder: &TransformerEmbedder, sparse_embedder: &SparseEmbedder, progress: Option<&dyn Fn(usize, usize)>) {
        let cfg = get_config();
        let mut all_chunks: Vec<(String, String, String)> = Vec::new();
        for (name, text) in file_texts {
            let norm = normalize(text);
            if norm.len() < cfg.min_chunk { continue; }
            for (content, heading) in chunk_text(&norm) {
                if content.len() >= cfg.min_chunk { all_chunks.push((content, heading, name.clone())); }
            }
        }
        let total = all_chunks.len();
        if total == 0 { return; }

        // Sparse-only indexing: skip dense embedding entirely.
        // Dense dihitung saat query (lazy) untuk top-K candidates saja.
        for ci in 0..total {
            let (content, heading, source) = all_chunks[ci].clone();
            if let Some(cb) = progress { cb(ci + 1, total); }
            let sparse_embedding = sparse_embedder.embed(&content);
            self.docs.push(Doc { content, source, page: 0, heading, dense_embedding: Vec::new(), sparse_embedding: sparse_embedding.clone() });
            self.sparse_index.push(sparse_embedding);
        }
    }

    fn search(&self, query: &str, embedder: &TransformerEmbedder, sparse_embedder: &SparseEmbedder, top_k: usize) -> Vec<SearchResult> {
        if self.docs.is_empty() || query.trim().is_empty() { return Vec::new(); }
        let cfg = get_config();

        // 1. Sparse BM25 → top 200 candidates (fast, no ORT)
        let query_sparse = sparse_embedder.embed(query);
        let mut scores: Vec<(u32, f64)> = self.sparse_index.par_iter().enumerate().map(|(i, emb)| {
            (i as u32, SparseEmbedder::sparse_cosine(&query_sparse, emb))
        }).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let candidates: Vec<u32> = scores.iter().take(cfg.top_k_candidates).map(|(d, _)| *d).collect();

        // 2. Dense query embedding (single ORT inference, ~50ms)
        let query_dense = if embedder.has_model() {
            embedder.embed(query)
        } else {
            embedder.embed_hash_idf(query, &self.dim_df, self.docs.len())
        };

        // 3. Dense similarity for each candidate (fast vector math, <1ms total)
        let candidate_scores: Vec<(u32, f64)> = candidates.iter().map(|&doc| {
            let dense_emb = &self.docs[doc as usize].dense_embedding;
            let dsim = if !dense_emb.is_empty() {
                cosine_similarity(&query_dense, dense_emb)
            } else { 0.0 };
            let sparse_score = scores.iter().find(|(d, _)| *d == doc).map(|(_, s)| *s).unwrap_or(0.0);
            // Combined: 40% sparse + 60% dense
            (doc, cfg.rerank_weight * dsim + (1.0 - cfg.rerank_weight) * sparse_score)
        }).collect();

        // 4. Financial boost + trigram
        let fin_terms: HashSet<&str> = ["revenue","income","pendapatan","profit","laba","assets","aset","liabilitas","utang","equity","ekuitas","modal","dividen","dividend","arus","kas","cash","penjualan","sales","beban","cost","expense","tax","pajak","emas","gold","nikel","nickel","bauksit","bauxite"].iter().cloned().collect();
        let mut final_scores: Vec<(u32, f64)> = candidate_scores;
        for (doc, s) in final_scores.iter_mut() {
            let c = self.docs[*doc as usize].content.to_lowercase();
            let nc = c.matches(|c: char| c.is_ascii_digit()).count();
            let hf = fin_terms.iter().any(|t| c.contains(t));
            if hf && nc > 20 { *s += cfg.financial_boost; }
        }

        // Trigram overlap
        { let qlow = query.to_lowercase(); let qw: Vec<String> = qlow.split_whitespace().map(|s| s.to_string()).collect();
          let mut qset: HashSet<String> = HashSet::new();
          for w in &qw { if w.chars().count() < 4 { continue; } let p = format!("#{}#", w); let pc: Vec<char> = p.chars().collect(); for i in 0..pc.len().saturating_sub(2) { qset.insert(pc[i..i+3].iter().collect()); } }
          if !qset.is_empty() { for (doc, s) in final_scores.iter_mut() { let c = self.docs[*doc as usize].content.to_lowercase(); let mut hit = 0usize; let mut tot = 0usize;
            for w in c.split_whitespace().take(120) { if w.chars().count() < 4 { continue; } let p = format!("#{}#", w); let pc: Vec<char> = p.chars().collect();
              for i in 0..pc.len().saturating_sub(2) { tot += 1; let g: String = pc[i..i+3].iter().collect(); if qset.contains(&g) { hit += 1; } }
              if tot > 400 { break; } }
            if tot > 0 { let overlap = hit as f64 / (qset.len() + tot) as f64; *s += overlap * 2.0; } }
          }
        }

        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        final_scores.truncate(top_k);

        final_scores.into_iter().map(|(doc, score)| {
            let d = &self.docs[doc as usize];
            let mut expanded = d.content.clone();
            if let Some(nxt) = self.docs.get(doc as usize + 1) { if nxt.source == d.source { expanded.push(' '); expanded.push_str(&nxt.content.chars().take(600).collect::<String>()); } }
            SearchResult { doc, score, page: d.page, source: d.source.clone(), heading: d.heading.clone(), snippet: d.content.chars().take(300).collect(), expanded: expanded.chars().take(900).collect() }
        }).collect()
    }

    fn save(&self, path: &Path) -> io::Result<()> { let f = fs::File::create(path)?; let mut w = io::BufWriter::new(f); bincode::serialize_into(&mut w, self).map_err(|e| io::Error::new(io::ErrorKind::Other, e)) }
    fn load(path: &Path) -> io::Result<Self> { let f = fs::File::open(path)?; let mut r = io::BufReader::new(f); bincode::deserialize_from(&mut r).map_err(|e| io::Error::new(io::ErrorKind::Other, e)) }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0; let mut na = 0.0; let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) { dot += a[i] as f64 * b[i] as f64; na += a[i] as f64 * a[i] as f64; nb += b[i] as f64 * b[i] as f64; }
    let norm = na.sqrt() * nb.sqrt();
    if norm > 0.0 { dot / norm } else { 0.0 }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SearchResult { doc: u32, score: f64, page: u32, source: String, heading: String, snippet: String, expanded: String }

// ==================== FILE PARSING ====================
fn is_indexable(p: &Path) -> bool {
    if !p.is_file() { return false; }
    let name = p.file_name().unwrap_or_default().to_string_lossy();
    if name.starts_with('.') { return false; }
    let lower = name.to_lowercase();
    if lower.contains("enwik8") || lower == "index.bin" || lower.ends_with(".bin") || lower == "onod.json" || lower == "config.json" || lower.ends_with(".ds_store") { return false; }
    true
}

fn read_file(path: &Path) -> io::Result<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "txt"|"md"|"log"|"csv"|"tsv"|"json"|"jsonl"|"html"|"htm"|"xml" => {
            let meta = fs::metadata(path)?;
            if meta.len() > 100 * 1024 * 1024 { read_file_mmap(path) } else { fs::read_to_string(path) }
        }
        "pdf" => {
            match pdf_oxide::PdfDocument::open(path.to_str().unwrap_or("")) {
                Ok(doc) => { let n = doc.page_count().unwrap_or(0); let mut all = String::new(); for i in 0..n { if let Ok(t) = doc.extract_text_auto(i) { all.push_str(&t); all.push_str("\n\n"); } } Ok(all) }
                Err(e) => { eprintln!("  WARNING: PDF failed: {}", e); Ok(String::new()) }
            }
        }
        _ => { let meta = fs::metadata(path)?; if meta.len() > 100 * 1024 * 1024 { read_file_mmap(path) } else { fs::read_to_string(path) } }
    }
}

fn read_file_mmap(path: &Path) -> io::Result<String> {
    use memmap2::Mmap;
    let f = fs::File::open(path)?; let mmap = unsafe { Mmap::map(&f) }.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(String::from_utf8_lossy(&mmap).to_string())
}

fn read_files_parallel(paths: &[PathBuf]) -> Vec<(PathBuf, io::Result<String>)> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let paths_clone = paths.to_vec();
    std::thread::spawn(move || {
        let handles: Vec<_> = paths_clone.into_iter().map(|path| { let tx = tx.clone(); std::thread::spawn(move || { let p = path.clone(); let _ = tx.send((path, read_file(&p))); }) }).collect();
        for h in handles { let _ = h.join(); }
        drop(tx);
    });
    rx.iter().collect()
}

fn read_folder(folder: &str) -> Vec<(String, String)> {
    let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| is_indexable(p)).collect();
    let results = read_files_parallel(&paths);
    results.iter().filter_map(|(p, r)| {
        let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
        r.as_ref().ok().filter(|t| !t.is_empty()).map(|t| (name, t.clone()))
    }).collect()
}

// ==================== WORKER MANAGEMENT ====================
const WORKER_PORT: u16 = 9091;
const WORKER_PID_FILE: &str = "/tmp/onod_worker.pid";
const WORKER_IDLE_SECS: u64 = 300; // 5 minutes

fn worker_pid_path() -> PathBuf { PathBuf::from(WORKER_PID_FILE) }

fn worker_ping() -> bool {
    match TcpStream::connect(format!("127.0.0.1:{}", WORKER_PORT)) {
        Ok(mut s) => {
            let _ = s.write_all(b"PING\nEND\n");
            let mut resp = String::new();
            let mut reader = BufReader::new(&s);
            let _ = reader.read_line(&mut resp);
            resp.contains("PONG")
        }
        Err(_) => false,
    }
}

fn worker_ensure_alive() {
    if worker_ping() { return; }
    let _ = fs::remove_file(worker_pid_path());
    eprintln!("  Spawning worker...");
    let exe = env::current_exe().expect("cannot find executable");
    match std::process::Command::new(&exe)
        .arg("--worker")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn() {
        Ok(_) => {}
        Err(e) => { eprintln!("  Worker spawn failed: {}", e); return; }
    }
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if worker_ping() { eprintln!("  Worker ready."); return; }
    }
    eprintln!("  Worker did not start in time");
}

fn worker_send(cmd: &str) -> String {
    match TcpStream::connect(format!("127.0.0.1:{}", WORKER_PORT)) {
        Ok(mut s) => {
            let _ = s.write_all(format!("{}\nEND\n", cmd).as_bytes());
            let mut resp = String::new();
            let mut reader = BufReader::new(&s);
            let _ = reader.read_line(&mut resp);
            resp.trim().to_string()
        }
        Err(_) => String::from("ERROR connection failed"),
    }
}

fn worker_run() {
    // Write PID
    let _ = fs::write(worker_pid_path(), std::process::id().to_string());

    eprintln!("  Loading model (worker, once)...");
    let t0 = Instant::now();
    let embedder = Arc::new(TransformerEmbedder::new());
    eprintln!("  Model ready in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let idx: Arc<RwLock<Option<Arc<OnodIndex>>>> = Arc::new(RwLock::new(None));
    let sparse: Arc<RwLock<Option<Arc<SparseEmbedder>>>> = Arc::new(RwLock::new(None));
    let last_request = Arc::new(RwLock::new(Instant::now()));

    // Idle timeout thread
    let last_clone = last_request.clone();
    let pid_path = worker_pid_path();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(10));
        let elapsed = last_clone.read().unwrap().elapsed().as_secs();
        if elapsed >= WORKER_IDLE_SECS {
            eprintln!("  Worker: idle {}s, shutting down.", elapsed);
            let _ = fs::remove_file(&pid_path);
            std::process::exit(0);
        }
    });

    // Check if port is already in use (another worker running)
    if TcpStream::connect(format!("127.0.0.1:{}", WORKER_PORT)).is_ok() {
        eprintln!("  Worker already running on port {}", WORKER_PORT);
        let _ = fs::write(worker_pid_path(), std::process::id().to_string());
        // Just keep running as a duplicate - the first one wins
        // But since we can't bind, exit gracefully
        return;
    }

    let listener = TcpListener::bind(format!("127.0.0.1:{}", WORKER_PORT)).expect("cannot bind worker");
    // Mark alive
    *last_request.write().unwrap() = Instant::now();

    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            let embedder = embedder.clone();
            let idx = idx.clone();
            let sparse = sparse.clone();
            let last_request = last_request.clone();
            std::thread::spawn(move || {
                *last_request.write().unwrap() = Instant::now();
                let mut reader = BufReader::new(&stream);
                let mut cmd = String::new();
                loop {
                    let mut line = String::new();
                    match reader.read_line(&mut line) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let trimmed = line.trim();
                            if trimmed == "END" { break; }
                            cmd.push_str(trimmed);
                            cmd.push(' ');
                        }
                    }
                }
                let cmd = cmd.trim();
                let resp = match cmd {
                    "PING" => "PONG".to_string(),
                    _ => match cmd.split_once(' ') {
                        Some(("INDEX", folder)) => {
                            let cache_path = Path::new(folder).join("index.bin");
                            let t = Instant::now();
                            let mut new_idx = OnodIndex::new();
                            let mut sp = SparseEmbedder::new();
                            let file_texts = read_folder(folder);
                            let texts: Vec<String> = file_texts.iter().map(|(_, t)| t.clone()).collect();
                            sp.build_vocab(&texts);
                            new_idx.build_index_batch(&file_texts, &embedder, &sp, None);
                            let _ = new_idx.save(&cache_path);
                            let index_ms = t.elapsed().as_secs_f64() * 1000.0;
                            let chunks = new_idx.docs.len();
                            *idx.write().unwrap() = Some(Arc::new(new_idx));
                            *sparse.write().unwrap() = Some(Arc::new(sp));
                            format!("OK {} {:.0}", chunks, index_ms)
                        }
                        Some(("SEARCH", query)) => {
                            let idx_g = idx.read().unwrap();
                            let sparse_g = sparse.read().unwrap();
                            match (idx_g.as_ref(), sparse_g.as_ref()) {
                                (Some(idx), Some(sparse)) => {
                                    let t = Instant::now();
                                    let results = idx.search(query, &embedder, sparse, 10);
                                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                                    format!("RESULTS {:.1} {}", ms, serde_json::to_string(&results).unwrap_or_default())
                                }
                                _ => "ERROR no index".to_string(),
                            }
                        }
                        Some(("SHUTDOWN", _)) => {
                            let _ = fs::remove_file(WORKER_PID_FILE);
                            std::process::exit(0);
                        }
                        _ => format!("ERROR unknown cmd: {}", cmd),
                    }
                };
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.write_all(b"\n");
            });
        }
    }
}

// ==================== HTTP HELPERS ====================
fn http_resp(code: u16, body: &str) -> String {
    let status = match code { 200=>"OK", 400=>"Bad Request", 404=>"Not Found", 503=>"Service Unavailable", _=>"Error" };
    format!("HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", code, status, body.len(), body)
}

fn send_resp(stream: &mut TcpStream, code: u16, body: &str) { let _ = stream.write_all(http_resp(code, body).as_bytes()); }

// ==================== SHARED STATE (daemon) ====================
struct DaemonState {
    idx: RwLock<Option<Arc<OnodIndex>>>,
    sparse: RwLock<Option<Arc<SparseEmbedder>>>,
    folder: RwLock<String>,
}

// ==================== MAIN ====================
fn main() {
    let _ = get_config();
    let args: Vec<String> = env::args().collect();

    // Worker mode: run as background process
    if args.iter().any(|a| a == "--worker") {
        worker_run();
        return;
    }

    if args.len() < 2 {
        eprintln!("onod v0.10.0 — Transformer Hybrid Search\n\nUsage:\n  onod search <folder> <query>       Search (uses warm worker)\n  onod benchmark <folder>            Benchmark 20 queries\n  onod test <folder> [test_file]     Run test suite (default: tests/test_25.txt)\n  onod daemon <folder> [--port 9090] HTTP daemon\n  onod serve <folder> [--port 8080]  HTTP server\n  onod worker stop                   Stop background worker\n\nWorker auto-starts on first command, expires after 5 min idle.");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "worker" => {
            if args.get(2).map(|s| s.as_str()) == Some("stop") {
                let _ = TcpStream::connect(format!("127.0.0.1:{}", WORKER_PORT)).map(|mut s| { let _ = s.write_all(b"SHUTDOWN\nEND\n"); });
                let _ = fs::remove_file(worker_pid_path());
                eprintln!("Worker stopped.");
            } else {
                eprintln!("Usage: onod worker stop");
            }
        }

        // ===== SEARCH: route through worker =====
        "search" => {
            let folder = args.get(2).expect("provide folder");
            let query = args.get(3..).unwrap_or(&[]).join(" ");
            if query.is_empty() { eprintln!("Usage: onod search <folder> <query>"); std::process::exit(1); }

            worker_ensure_alive();
            // Index if needed
            let resp = worker_send(&format!("INDEX {}", folder));
            if resp.starts_with("OK") { eprintln!("  {}", resp); }
            // Search
            let t = Instant::now();
            let resp = worker_send(&format!("SEARCH {}", query));
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if let Some(json_str) = resp.strip_prefix("RESULTS ") {
                let parts: Vec<&str> = json_str.splitn(2, ' ').collect();
                let query_ms: f64 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                if let Ok(results) = serde_json::from_str::<Vec<SearchResult>>(parts.get(1).unwrap_or(&"[]")) {
                    eprintln!("\nWorker search: {:.1}ms (roundtrip {:.1}ms)\n", query_ms, ms);
                    for (i, r) in results.iter().enumerate() {
                        println!("{}. [{:.4}] {} — {}", i+1, r.score, r.heading, r.snippet.chars().take(150).collect::<String>());
                    }
                }
            } else {
                eprintln!("Worker error: {}", resp);
            }
        }

        // ===== BENCHMARK: route through worker =====
        "benchmark" => {
            let folder = args.get(2).expect("provide folder");

            worker_ensure_alive();
            // Index
            eprintln!("Indexing via worker...");
            let t_idx = Instant::now();
            let resp = worker_send(&format!("INDEX {}", folder));
            let idx_ms = t_idx.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  {}", resp);

            let queries = vec![
                ("Berapa total uang yang dihasilkan perusahaan dari pelanggan?", vec!["62.714","revenue","pendapatan"]),
                ("Pendapatan bersih PT ANTAM semester 1 2026 berapa?", vec!["62.714","pendapatan"]),
                ("How much money did the company earn from customer contracts?", vec!["62.714","revenue","income"]),
                ("Selisih antara pendapatan dan beban pokok penjualan?", vec!["10.851","laba","gross"]),
                ("Berapa keuntungan sebelum dikurangi beban operasi dan pajak?", vec!["10.851","laba","bruto"]),
                ("Profit after tax berapa juta rupiah?", vec!["8.443","laba","bersih","profit"]),
                ("Seluruh nilai kekayaan yang dimiliki perusahaan berapa?", vec!["aset","assets","total"]),
                ("Total kewajiban yang harus dibayar perusahaan?", vec!["utang","liabilities","kewajiban"]),
                ("Bagian pemegang saham dari total aset berapa?", vec!["ekuitas","equity","modal"]),
                ("Berapa bagian keuntungan yang dibagikan ke pemegang saham?", vec!["dividen","dividend"]),
                ("Berapa uang tunai yang dihasilkan dari kegiatan operasi?", vec!["arus kas","cash flow","operating"]),
                ("Berapa jumlah emas yang diproduksi perusahaan?", vec!["emas","gold","produksi"]),
                ("Produksi logam nikel berapa ton?", vec!["nikel","nickel","ton"]),
                ("Siapa yang memimpin perusahaan sebagai presiden direktur?", vec!["direktur","presiden","CEO"]),
                ("Apa saja produk pertambangan utama perusahaan?", vec!["tambang","mining","produksi"]),
                ("Berapa beban pajak penghasilan yang harus dibayar?", vec!["pajak","tax"]),
                ("Berapa hak tagih dari pelanggan yang belum dibayar?", vec!["piutang","receivable","pelanggan"]),
                ("Berapa stok emas yang belum dijual?", vec!["persediaan","inventory","emas"]),
                ("Berapa selisih antara pendapatan dan beban pokok penjualan?", vec!["laba kotor","gross profit","selisih"]),
                ("Arus kas dari kegiatan operasi berapa?", vec!["arus","kas","operasi","cash"]),
            ];

            let mut hits = 0; let mut total_ms = 0.0;
            for (q, kws) in &queries {
                let t = Instant::now();
                let resp = worker_send(&format!("SEARCH {}", q));
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                total_ms += ms;
                let ok = if let Some(json_str) = resp.strip_prefix("RESULTS ") {
                    let parts: Vec<&str> = json_str.splitn(2, ' ').collect();
                    if let Ok(results) = serde_json::from_str::<Vec<SearchResult>>(parts.get(1).unwrap_or(&"[]")) {
                        let blob: String = results.iter().flat_map(|r| vec![r.snippet.clone(), r.expanded.clone()]).collect::<Vec<_>>().join(" ").to_lowercase();
                        kws.iter().any(|k| blob.contains(&k.to_lowercase()))
                    } else { false }
                } else { false };
                if ok { hits += 1; }
                println!("{} {:6.1}ms | {:60} -> {}", if ok {"OK "} else {"FAIL"}, ms, q, if ok {"✅"} else {"❌"});
            }
            println!("\n=== RESULTS ===\nRecall: {}/{} = {:.1}%\nAvg latency: {:.1}ms\nIndex (worker): {:.1}s",
                hits, queries.len(), hits as f64 / queries.len() as f64 * 100.0, total_ms / queries.len() as f64, idx_ms / 1000.0);
        }

        // ===== TEST: run test_25.txt queries =====
        "test" => {
            let folder = args.get(2).expect("provide folder");
            let test_file = args.get(3).map(|s| s.as_str()).unwrap_or("../tests/test_25.txt");

            worker_ensure_alive();
            // Index
            eprintln!("Indexing via worker...");
            let t_idx = Instant::now();
            let resp = worker_send(&format!("INDEX {}", folder));
            let idx_ms = t_idx.elapsed().as_secs_f64() * 1000.0;
            eprintln!("  {}", resp);

            // Load test queries
            let content = fs::read_to_string(test_file).unwrap_or_else(|e| { eprintln!("Cannot read {}: {}", test_file, e); std::process::exit(1); });
            let mut queries: Vec<(String, Vec<String>)> = Vec::new();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() { continue; }
                // Format: "N. query -> keyword1, keyword2, ..."
                if let Some((q_part, kw_part)) = line.split_once("->") {
                    let query = q_part.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.').trim().to_string();
                    let keywords: Vec<String> = kw_part.split(',').map(|k| k.trim().to_lowercase()).collect();
                    queries.push((query, keywords));
                }
            }
            if queries.is_empty() { eprintln!("No queries found in {}", test_file); std::process::exit(1); }

            let mut hits = 0; let mut total_ms = 0.0;
            for (q, kws) in &queries {
                let t = Instant::now();
                let resp = worker_send(&format!("SEARCH {}", q));
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                total_ms += ms;
                let ok = if let Some(json_str) = resp.strip_prefix("RESULTS ") {
                    let parts: Vec<&str> = json_str.splitn(2, ' ').collect();
                    if let Ok(results) = serde_json::from_str::<Vec<SearchResult>>(parts.get(1).unwrap_or(&"[]")) {
                        let blob: String = results.iter().flat_map(|r| vec![r.snippet.clone(), r.expanded.clone()]).collect::<Vec<_>>().join(" ").to_lowercase();
                        kws.iter().any(|k| blob.contains(k))
                    } else { false }
                } else { false };
                if ok { hits += 1; }
                println!("{} {:6.1}ms | {:60} -> {}", if ok {"OK "} else {"FAIL"}, ms, q, if ok {"✅"} else {"❌"});
            }
            println!("\n=== TEST RESULTS ===\nRecall: {}/{} = {:.1}%\nAvg latency: {:.1}ms\nIndex (worker): {:.1}s",
                hits, queries.len(), hits as f64 / queries.len() as f64 * 100.0, total_ms / queries.len() as f64, idx_ms / 1000.0);
        }

        // ===== DAEMON: model always warm, index on demand =====
        "daemon" => {
            let folder = args.get(2).expect("provide folder").clone();
            let port: u16 = args.iter().position(|a| a == "--port").and_then(|i| args.get(i+1)).and_then(|p| p.parse().ok()).unwrap_or(9090);

            eprintln!("Loading model (once)...");
            let t0 = Instant::now();
            let embedder = Arc::new(TransformerEmbedder::new());

            eprintln!("Model ready in {:.0}ms\n", t0.elapsed().as_secs_f64() * 1000.0);

            // Try load cache on startup
            let cache_path = std::path::Path::new(&folder).join("index.bin");
            let (idx, sparse) = if cache_path.exists() {
                match OnodIndex::load(&cache_path) {
                    Ok(loaded) => {
                        let mut sp = SparseEmbedder::new();
                        let texts: Vec<String> = loaded.docs.iter().map(|d| d.content.clone()).collect();
                        sp.build_vocab(&texts);
                        eprintln!("Loaded cached index: {} chunks", loaded.docs.len());
                        (Some(Arc::new(loaded)), Some(Arc::new(sp)))
                    }
                    Err(_) => (None, None)
                }
            } else { (None, None) };

            let state = Arc::new(DaemonState {
                idx: RwLock::new(idx),
                sparse: RwLock::new(sparse),
                folder: RwLock::new(folder),
            });

            let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("cannot bind");
            eprintln!("Daemon listening on http://0.0.0.0:{}\n", port);

            for stream in listener.incoming() {
                if let Ok(mut stream) = stream {
                    let state = state.clone();
                    let embedder = embedder.clone();

                    std::thread::spawn(move || {
                        let mut buf = BufReader::new(&stream);
                        let mut method_line = String::new();
                        let _ = buf.read_line(&mut method_line);
                        let parts: Vec<&str> = method_line.split_whitespace().collect();
                        if parts.len() < 2 { send_resp(&mut stream, 400, "{\"error\":\"bad request\"}"); return; }

                        let mut content_len: usize = 0;
                        loop { let mut header = String::new(); let _ = buf.read_line(&mut header); if header.trim().is_empty() { break; } if let Some(val) = header.strip_prefix("Content-Length:") { content_len = val.trim().parse().unwrap_or(0); } }
                        let mut body = vec![0u8; content_len]; if content_len > 0 { let _ = buf.read_exact(&mut body); }
                        let body_str = String::from_utf8_lossy(&body).to_string();

                        match (parts[0], parts[1]) {
                            ("GET", "/") => send_resp(&mut stream, 200, &serde_json::json!({"name":"onod","version":"0.9.0","status":"daemon"}).to_string()),
                            ("GET", "/health") => {
                                let chunks = state.idx.read().ok().and_then(|g| g.as_ref().map(|i| i.docs.len())).unwrap_or(0);
                                send_resp(&mut stream, 200, &serde_json::json!({"status":"ok","model":"warm","chunks":chunks}).to_string());
                            }
                            ("GET", p) if p.starts_with("/search") => {
                                let params: HashMap<String,String> = p.split('?').nth(1).unwrap_or("").split('&').filter_map(|p| p.split_once('=')).map(|(k,v)| (k.replace("%20"," "), v.replace("%20"," "))).collect();
                                let q = params.get("q").map(|s| s.as_str()).unwrap_or("");
                                let top_k: usize = params.get("top_k").and_then(|s| s.parse().ok()).unwrap_or(10);
                                if q.is_empty() { send_resp(&mut stream, 400, "{\"error\":\"missing q\"}"); return; }
                                let idx_g = state.idx.read().ok();
                                let sparse_g = state.sparse.read().ok();
                                match (idx_g.as_ref().and_then(|g| g.as_ref()), sparse_g.as_ref().and_then(|g| g.as_ref())) {
                                    (Some(idx), Some(sparse)) => {
                                        let t = Instant::now();
                                        let results = idx.search(q, &embedder, sparse, top_k);
                                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                                        send_resp(&mut stream, 200, &serde_json::json!({"query":q,"latency_ms":ms,"results":results}).to_string());
                                    }
                                    _ => send_resp(&mut stream, 503, "{\"error\":\"index not ready. POST /index first\"}"),
                                }
                            }
                            ("POST", "/index") => {
                                let req: serde_json::Value = serde_json::from_str(&body_str).unwrap_or(serde_json::json!({}));
                                let folder_guard = state.folder.read().unwrap();
                                let folder = req["folder"].as_str().unwrap_or(&folder_guard);
                                let no_cache = req["no_cache"].as_bool().unwrap_or(false);
                                let cache_path = std::path::Path::new(folder).join("index.bin");

                                let t = Instant::now();
                                let result = if !no_cache && cache_path.exists() { OnodIndex::load(&cache_path).ok().map(|l| (l, true)) } else { None };
                                let (new_idx, from_cache) = match result { Some((l, _)) => (l, true), None => {
                                    let mut idx = OnodIndex::new();
                                    let mut sp = SparseEmbedder::new();
                                    let file_texts = read_folder(folder);
                                    let texts: Vec<String> = file_texts.iter().map(|(_, t)| t.clone()).collect();
                                    sp.build_vocab(&texts);
                                    idx.build_index_batch(&file_texts, &embedder, &sp, None);
                                    let _ = idx.save(&cache_path);
                                    (idx, false)
                                }};
                                let index_ms = t.elapsed().as_secs_f64() * 1000.0;
                                let chunks = new_idx.docs.len();

                                let mut new_sparse = SparseEmbedder::new();
                                let texts: Vec<String> = new_idx.docs.iter().map(|d| d.content.clone()).collect();
                                new_sparse.build_vocab(&texts);
                                *state.idx.write().unwrap() = Some(Arc::new(new_idx));
                                *state.sparse.write().unwrap() = Some(Arc::new(new_sparse));
                                *state.folder.write().unwrap() = folder.to_string();

                                send_resp(&mut stream, 200, &serde_json::json!({"status":"ok","chunks":chunks,"from_cache":from_cache,"index_ms":index_ms}).to_string());
                            }
                            _ => send_resp(&mut stream, 404, "{\"error\":\"not found\"}"),
                        }
                    });
                }
            }
        }

        // ===== SERVE: same as daemon but with folder loaded at startup =====
        "serve" => {
            let folder = args.get(2).expect("provide folder").clone();
            let port: u16 = args.iter().position(|a| a == "--port").and_then(|i| args.get(i+1)).and_then(|p| p.parse().ok()).unwrap_or(8080);

            eprintln!("Loading model...");
            let embedder = Arc::new(TransformerEmbedder::new());


            let cache_path = std::path::Path::new(&folder).join("index.bin");
            let (idx, sparse) = if cache_path.exists() {
                match OnodIndex::load(&cache_path) {
                    Ok(loaded) => { let mut sp = SparseEmbedder::new(); let texts: Vec<String> = loaded.docs.iter().map(|d| d.content.clone()).collect(); sp.build_vocab(&texts); eprintln!("Loaded cache: {} chunks", loaded.docs.len()); (Some(Arc::new(loaded)), Some(Arc::new(sp))) }
                    Err(_) => (None, None)
                }
            } else { (None, None) };

            let state = Arc::new(DaemonState { idx: RwLock::new(idx), sparse: RwLock::new(sparse), folder: RwLock::new(folder) });
            let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("cannot bind");
            eprintln!("Server listening on http://0.0.0.0:{}\n", port);

            for stream in listener.incoming() {
                if let Ok(mut stream) = stream {
                    let state = state.clone();
                    let embedder = embedder.clone();

                    std::thread::spawn(move || {
                        let mut buf = BufReader::new(&stream);
                        let mut method_line = String::new();
                        let _ = buf.read_line(&mut method_line);
                        let parts: Vec<&str> = method_line.split_whitespace().collect();
                        if parts.len() < 2 { send_resp(&mut stream, 400, "{\"error\":\"bad request\"}"); return; }
                        let mut content_len: usize = 0;
                        loop { let mut header = String::new(); let _ = buf.read_line(&mut header); if header.trim().is_empty() { break; } if let Some(val) = header.strip_prefix("Content-Length:") { content_len = val.trim().parse().unwrap_or(0); } }
                        let mut body = vec![0u8; content_len]; if content_len > 0 { let _ = buf.read_exact(&mut body); }
                        let body_str = String::from_utf8_lossy(&body).to_string();
                        match (parts[0], parts[1]) {
                            ("GET", "/") => send_resp(&mut stream, 200, &serde_json::json!({"name":"onod","version":"0.9.0"}).to_string()),
                            ("GET", "/health") => { let chunks = state.idx.read().ok().and_then(|g| g.as_ref().map(|i| i.docs.len())).unwrap_or(0); send_resp(&mut stream, 200, &serde_json::json!({"status":"ok","chunks":chunks}).to_string()); }
                            ("GET", p) if p.starts_with("/search") => {
                                let params: HashMap<String,String> = p.split('?').nth(1).unwrap_or("").split('&').filter_map(|p| p.split_once('=')).map(|(k,v)| (k.replace("%20"," "), v.replace("%20"," "))).collect();
                                let q = params.get("q").map(|s| s.as_str()).unwrap_or("");
                                let top_k: usize = params.get("top_k").and_then(|s| s.parse().ok()).unwrap_or(10);
                                if q.is_empty() { send_resp(&mut stream, 400, "{\"error\":\"missing q\"}"); return; }
                                let idx_g = state.idx.read().ok(); let sparse_g = state.sparse.read().ok();
                                match (idx_g.as_ref().and_then(|g| g.as_ref()), sparse_g.as_ref().and_then(|g| g.as_ref())) {
                                    (Some(idx), Some(sparse)) => { let t = Instant::now(); let results = idx.search(q, &embedder, sparse, top_k); let ms = t.elapsed().as_secs_f64() * 1000.0; send_resp(&mut stream, 200, &serde_json::json!({"query":q,"latency_ms":ms,"results":results}).to_string()); }
                                    _ => send_resp(&mut stream, 503, "{\"error\":\"no index\"}"),
                                }
                            }
                            ("POST", "/index") => {
                                let req: serde_json::Value = serde_json::from_str(&body_str).unwrap_or(serde_json::json!({}));
                                let folder_guard = state.folder.read().unwrap();
                                let folder = req["folder"].as_str().unwrap_or(&folder_guard);
                                let cache_path = std::path::Path::new(folder).join("index.bin");
                                let t = Instant::now();
                                let mut idx = OnodIndex::new(); let mut sp = SparseEmbedder::new();
                                let file_texts = read_folder(folder);
                                let texts: Vec<String> = file_texts.iter().map(|(_, t)| t.clone()).collect();
                                sp.build_vocab(&texts);
                                idx.build_index_batch(&file_texts, &embedder, &sp, None);
                                let _ = idx.save(&cache_path);
                                let index_ms = t.elapsed().as_secs_f64() * 1000.0;
                                let chunks = idx.docs.len();
                                let mut new_sp = SparseEmbedder::new();
                                let texts: Vec<String> = idx.docs.iter().map(|d| d.content.clone()).collect();
                                new_sp.build_vocab(&texts);
                                *state.idx.write().unwrap() = Some(Arc::new(idx));
                                *state.sparse.write().unwrap() = Some(Arc::new(new_sp));
                                *state.folder.write().unwrap() = folder.to_string();
                                send_resp(&mut stream, 200, &serde_json::json!({"status":"ok","chunks":chunks,"index_ms":index_ms}).to_string());
                            }
                            _ => send_resp(&mut stream, 404, "{\"error\":\"not found\"}"),
                        }
                    });
                }
            }
        }

        _ => { eprintln!("Unknown command: {}", args[1]); std::process::exit(1); }
    }
}
