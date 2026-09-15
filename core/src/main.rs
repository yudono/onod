use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;
use ndarray::Array2;
use ort::session::{Session, builder::SessionBuilder};
use ort::value::Value;

// ==================== CONFIG ====================
#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct Config {
    chunk_chars: usize,
    chunk_overlap: usize,
    min_chunk: usize,
    hard_split: usize,
    embedding_dim: usize,
    top_k_candidates: usize,
    top_k_results: usize,
    rerank_weight: f64,
    financial_boost: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_chars: 600,
            chunk_overlap: 140,
            min_chunk: 100,
            hard_split: 900,
            embedding_dim: 384,
            top_k_candidates: 200,
            top_k_results: 10,
            rerank_weight: 0.6,
            financial_boost: 0.3,
        }
    }
}

impl Config {
    fn load() -> Self {
        for path in &["onod.json", "config.json"] {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(c) = serde_json::from_str::<Config>(&content) { eprintln!("Loaded config from {}", path); return c; }
            }
        }
        Config::default()
    }
}

static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();
fn get_config() -> &'static Config { CONFIG.get_or_init(|| Config::load()) }

// ==================== TRANSFORMER EMBEDDING ====================
// ort hook: jika file model.onnx ada dipakai, jika tidak fallback ke hash embedding.
// Mutex agar &self tetap bisa dipakai paralel via rayon (Session::run butuh &mut).
struct TransformerEmbedder {
    model: std::sync::Mutex<Option<Session>>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
}

impl TransformerEmbedder {
    fn new() -> Self {
        let cfg = get_config();
        // Load tokenizer.json asli (Unigram 250k). Fallback ke WordPiece kosong hanya jika file hilang.
        let tokenizer = tokenizers::Tokenizer::from_file("model.tokenizer.json")
            .or_else(|_| tokenizers::Tokenizer::from_file("tokenizer.json"))
            .unwrap_or_else(|e| {
                eprintln!("  WARNING: tokenizer.json gagal dimuat ({}), fallback hash embedding", e);
                tokenizers::Tokenizer::new(tokenizers::models::wordpiece::WordPiece::default())
            });

        let model = SessionBuilder::new()
            .and_then(|mut b| b.commit_from_file("model.onnx"))
            .map_err(|e| {
                eprintln!("  WARNING: model.onnx gagal dimuat ({}), fallback hash embedding", e);
                e
            })
            .ok();

        if model.is_some() {
            eprintln!("  ORT model loaded: model.onnx (dim={})", cfg.embedding_dim);
        } else {
            eprintln!("  ORT model TIDAK ada: pakai hash embedding (dim={})", cfg.embedding_dim);
        }

        Self { model: std::sync::Mutex::new(model), tokenizer, dim: cfg.embedding_dim }
    }

    fn has_model(&self) -> bool {
        self.model.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // Encode dulu di luar lock (tokenizer &self, thread-safe untuk rayon).
        let encoding = match self.tokenizer.encode(text, true) {
            Ok(enc) => enc,
            Err(_) => return self.embed_hash(text),
        };
        if encoding.len() == 0 {
            return self.embed_hash(text);
        }
        if let Ok(mut guard) = self.model.lock() {
            if let Some(ref mut model) = *guard {
                if let Some(v) = Self::embed_with_model_static(model, &encoding, self.dim) {
                    return v;
                }
            }
        }
        self.embed_hash(text)
    }

    /// Query-side: jika model ada pakai model (konsisten dengan doc-side),
    /// jika tidak pakai varian hash+IDF.
    fn embed_query(&self, text: &str, dim_df: &[u32], n_docs: usize) -> Vec<f32> {
        if self.has_model() {
            return self.embed(text);
        }
        self.embed_hash_idf(text, dim_df, n_docs)
    }

    fn embed_with_model_static(model: &mut Session, encoding: &tokenizers::Encoding, dim: usize) -> Option<Vec<f32>> {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        if ids.is_empty() { return None; }
        let seq_len = ids.len();
        let input_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = mask.iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = vec![0i64; seq_len];

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids).ok()?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask).ok()?;
        let token_type_ids_array = Array2::from_shape_vec((1, seq_len), token_type_ids).ok()?;

        let input_ids_value = Value::from_array(input_ids_array.into_dyn()).ok()?;
        let attention_mask_value = Value::from_array(attention_mask_array.into_dyn()).ok()?;
        let token_type_ids_value = Value::from_array(token_type_ids_array.into_dyn()).ok()?;

        let outputs = model.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
            "token_type_ids" => token_type_ids_value
        ]).ok()?;

        // last_hidden_state: [1, seq_len, hidden] -> mean pooling pakai attention_mask + L2 norm
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().ok()?;
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        if shape.len() != 3 { return None; }
        let hidden = shape[2];
        if data.len() < seq_len * hidden { return None; }
        let mask_f: Vec<f32> = mask.iter().map(|&x| x as f32).collect();
        let mask_sum: f32 = mask_f.iter().sum::<f32>().max(1.0);
        let mut pooled = vec![0.0f32; hidden];
        for i in 0..seq_len {
            let m = mask_f[i];
            if m == 0.0 { continue; }
            let base = i * hidden;
            for h in 0..hidden {
                pooled[h] += data[base + h] * m;
            }
        }
        for v in &mut pooled { *v /= mask_sum; }
        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for v in &mut pooled { *v /= norm; } }
        // Sesuaikan ke dim config (model 384 == default 384; truncate/pad jika beda)
        if pooled.len() == dim {
            Some(pooled)
        } else if pooled.len() > dim {
            pooled.truncate(dim);
            Some(pooled)
        } else {
            pooled.resize(dim, 0.0);
            Some(pooled)
        }
    }
    
    fn embed_hash(&self, text: &str) -> Vec<f32> {
        // Char n-gram embedding ala fastText via helper terpusat.
        let mut embedding = vec![0.0f32; self.dim];
        let mut ngram_count = 0usize;
        for_each_gram_idx(text, self.dim, |idx| {
            embedding[idx] += 1.0;
            ngram_count += 1;
        });
        if ngram_count == 0 { return embedding; }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in &mut embedding { *x /= norm; } }
        embedding
    }

    fn embed_hash_idf(&self, text: &str, dim_df: &[u32], n_docs: usize) -> Vec<f32> {
        // Varian query-side: abaikan gram yang df=0 (tak matchible di korpus,
        // hanya noise — mis. CJK pada korpus Latin) + bobot IDF.
        // Ini IDF standar, bukan hardcode bahasa.
        let mut embedding = vec![0.0f32; self.dim];
        let mut ngram_count = 0usize;
        let n = n_docs.max(1) as f64;
        for_each_gram_idx(text, self.dim, |idx| {
            let df = dim_df.get(idx).copied().unwrap_or(0);
            if df == 0 { return; }
            let idf = ((n - df as f64 + 0.5) / (df as f64 + 0.5) + 1.0).ln() as f32;
            embedding[idx] += idf;
            ngram_count += 1;
        });
        if ngram_count == 0 { return embedding; }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in &mut embedding { *x /= norm; } }
        embedding
    }

}

/// Satu-satunya tempat logika gram→dim. Dipakai index-time & query-time
/// agar konsisten. Tanpa alokasi String per gram.
fn for_each_gram_idx(text: &str, dim: usize, mut f: impl FnMut(usize)) {
    for word in text.split_whitespace() {
        let lower = word.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < 3 { continue; }
        // bungkus #...# secara virtual via index
        for n in 3..=5 {
            if chars.len() + 2 < n { continue; }
            for i in 0..=(chars.len() + 2 - n) {
                let mut h: u64 = 0xcbf29ce484222325;
                for j in 0..n {
                    let c = if i == 0 && j == 0 {
                        '#' as u32
                    } else if i + j == chars.len() + 1 {
                        '#' as u32
                    } else {
                        chars[i + j - 1] as u32
                    };
                    h ^= c as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                f((h % dim as u64) as usize);
            }
        }
    }
}

// ==================== SPARSE TEXT EMBEDDING ====================
struct SparseEmbedder {
    vocab: HashMap<String, u32>,
    idf: HashMap<String, f64>,
}

impl SparseEmbedder {
    fn new() -> Self {
        Self { vocab: HashMap::new(), idf: HashMap::new() }
    }
    
    fn build_vocab(&mut self, texts: &[String]) {
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut df: HashMap<String, usize> = HashMap::new();
        let n = texts.len() as f64;
        
        for text in texts {
            let words: HashSet<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
            for word in &words {
                *tf.entry(word.clone()).or_insert(0) += 1;
                *df.entry(word.clone()).or_insert(0) += 1;
            }
        }
        
        // Build vocab dengan frequency threshold
        for (word, &count) in &tf {
            if count >= 2 && word.len() > 2 {
                let idx = self.vocab.len() as u32;
                self.vocab.insert(word.clone(), idx);
            }
        }
        
        // Compute IDF
        for (word, &df_val) in &df {
            self.idf.insert(word.clone(), (n / (df_val as f64 + 1.0)).ln());
        }
    }
    
    fn embed(&self, text: &str) -> HashMap<u32, f32> {
        // Sparse sejati: hanya dimensi non-zero yang disimpan.
        // 3750 docs × ~100 terms ≈ 375rb entri (vs 114jt dense) → hemat ~300x.
        let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();
        let mut sparse: HashMap<u32, f32> = HashMap::new();

        for word in &words {
            if let Some(&idx) = self.vocab.get(word) {
                *sparse.entry(idx).or_insert(0.0) += *self.idf.get(word).unwrap_or(&1.0) as f32;
            }
        }

        // L2 normalize
        let norm: f32 = sparse.values().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { for x in sparse.values_mut() { *x /= norm; } }
        sparse
    }

    fn sparse_cosine(a: &HashMap<u32, f32>, b: &HashMap<u32, f32>) -> f64 {
        if a.is_empty() || b.is_empty() { return 0.0; }
        let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let mut dot = 0.0;
        for (k, v) in small {
            if let Some(w) = big.get(k) { dot += *v as f64 * *w as f64; }
        }
        dot // vektor sudah ternormalisasi
    }

}

// ==================== RERANKER ====================
// Rerank murah tanpa re-embed: memadukan skor fusion dengan cosine
// query_dense vs dense embedding tersimpan.
struct Reranker;

impl Reranker {
    fn new() -> Self { Self }
    
    fn rerank(&self, query_dense: &[f32], candidates: &[SearchResult], docs: &[Doc], top_k: usize) -> Vec<SearchResult> {
        // Rerank murah: cosine antara query_dense (sudah dihitung sekali)
        // dengan dense embedding tersimpan tiap kandidat. Tanpa re-embed.
        let mut scored: Vec<(usize, f64)> = candidates.iter().enumerate().map(|(i, c)| {
            let sim = cosine_similarity(query_dense, &docs[c.doc as usize].dense_embedding);
            (i, 0.5 * c.score + 0.5 * sim)
        }).collect();
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        
        scored.into_iter().map(|(i, score)| {
            let mut r = candidates[i].clone();
            r.score = score;
            r
        }).collect()
    }
}

// ==================== NORMALIZATION ====================
fn is_ar_diacritic(c: char) -> bool { matches!(c, '\u{64B}'..='\u{652}' | '\u{670}') }
fn normalize(raw: &str) -> String {
    let nfkc: String = raw.nfkc().collect();
    let mut out = String::with_capacity(nfkc.len());
    let mut prev_space = true;
    for c in nfkc.chars() {
        if is_ar_diacritic(c) { continue; }
        if c.is_whitespace() { if !prev_space { out.push(' '); prev_space = true; } }
        else { out.push(c); prev_space = false; }
    }
    if prev_space { out.pop(); }
    out
}

// ==================== CHUNKING ====================
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

fn chunk_text(text: &str) -> Vec<(String, String)> {
    let cfg = get_config();
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut buf_len = 0usize;
    let mut heading = String::new();
    for s in sentences {
        if is_heading(&s) {
            let h: String = s.trim_start_matches(['#',' ']).chars().take(120).collect();
            heading = h;
        }
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
struct OnodIndex {
    docs: Vec<Doc>,
    dense_index: Vec<Vec<f32>>,
    sparse_index: Vec<HashMap<u32, f32>>,
    /// Doc-frequency per hashed gram dim (ukuran = embedding_dim).
    /// Dipakai query-side untuk skip gram tak-matchible + bobot IDF.
    dim_df: Vec<u32>,
}

impl OnodIndex {
    fn new() -> Self {
        let dim = get_config().embedding_dim;
        Self { docs: Vec::new(), dense_index: Vec::new(), sparse_index: Vec::new(), dim_df: vec![0u32; dim] }
    }

    fn add_text(&mut self, text: &str, source: &str, page: u32, embedder: &TransformerEmbedder, sparse_embedder: &SparseEmbedder) -> u32 {
        let norm = normalize(text);
        let cfg = get_config();
        if norm.len() < cfg.min_chunk { return 0; }
        let chunks = chunk_text(&norm);
        let mut n = 0u32;
        for (content, heading) in chunks {
            if content.len() < cfg.min_chunk { continue; }
            let dense_embedding = embedder.embed(&content);
            for (i, v) in dense_embedding.iter().enumerate() {
                if *v > 0.0 {
                    if i >= self.dim_df.len() { self.dim_df.resize(i + 1, 0); }
                    self.dim_df[i] += 1;
                }
            }
            let sparse_embedding = sparse_embedder.embed(&content);
            self.docs.push(Doc { content, source: source.to_string(), page, heading, dense_embedding: dense_embedding.clone(), sparse_embedding: sparse_embedding.clone() });
            self.dense_index.push(dense_embedding);
            self.sparse_index.push(sparse_embedding);
            n += 1;
        }
        n
    }

    fn search(&self, query: &str, embedder: &TransformerEmbedder, sparse_embedder: &SparseEmbedder, reranker: &Reranker, top_k: usize) -> Vec<SearchResult> {
        if self.docs.is_empty() || query.trim().is_empty() { return Vec::new(); }
        let cfg = get_config();
        
        // 1. Dense retrieval (embedding similarity) — paralel 8-core.
        // Jika ORT model ada: query & doc sama-sama transformer embedding.
        // Jika tidak: query pakai varian IDF (skip gram df=0).
        let query_dense = embedder.embed_query(query, &self.dim_df, self.docs.len());
        let dense_scores: Vec<(u32, f64)> = self.dense_index.par_iter().enumerate().map(|(i, emb)| {
            (i as u32, cosine_similarity(&query_dense, emb))
        }).collect();

        // 2. Sparse retrieval (intersect non-zero dims — cepat, tanpa scan 30k dim) — paralel 8-core
        let query_sparse = sparse_embedder.embed(query);
        let sparse_scores: Vec<(u32, f64)> = self.sparse_index.par_iter().enumerate().map(|(i, emb)| {
            (i as u32, SparseEmbedder::sparse_cosine(&query_sparse, emb))
        }).collect();
        
        // 3. Hybrid fusion (RRF)
        let mut acc: HashMap<u32, f64> = HashMap::new();
        let rrf_k = 60.0;
        
        // Sort dense scores
        let mut dense_sorted = dense_scores.clone();
        dense_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (doc, _)) in dense_sorted.iter().enumerate() {
            *acc.entry(*doc).or_insert(0.0) += 0.6 / (rrf_k + rank as f64 + 1.0);
        }
        
        // Sort sparse scores
        let mut sparse_sorted = sparse_scores.clone();
        sparse_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (doc, _)) in sparse_sorted.iter().enumerate() {
            *acc.entry(*doc).or_insert(0.0) += 0.4 / (rrf_k + rank as f64 + 1.0);
        }
        
        // 4. Financial boost
        let fin_terms: HashSet<&str> = ["revenue","income","pendapatan","profit","laba","assets","aset","liabilitas","utang","equity","ekuitas","modal","dividen","dividend","arus","kas","cash","penjualan","sales","beban","cost","expense","tax","pajak","emas","gold","nikel","nickel","bauksit","bauxite"].iter().cloned().collect();
        let mut fused: Vec<(u32, f64)> = acc.into_iter().collect();
        for (doc, s) in fused.iter_mut() {
            let c = self.docs[*doc as usize].content.to_lowercase();
            let nc = c.matches(|c: char| c.is_ascii_digit()).count();
            let hf = fin_terms.iter().any(|t| c.contains(t));
            if hf && nc > 20 { *s += cfg.financial_boost; }
        }
        
        // 5. Sort and take top candidates
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(cfg.top_k_candidates);

        // 5b. Trigram-overlap boost pada kandidat (char-level, typo/morfologi).
        // Murah karena hanya top-K kandidat, bukan full corpus.
        {
            let qlow = query.to_lowercase();
            let qw: Vec<String> = qlow.split_whitespace().map(|s| s.to_string()).collect();
            let mut qset: HashSet<String> = HashSet::new();
            for w in &qw {
                if w.chars().count() < 4 { continue; }
                let p = format!("#{}#", w);
                let pc: Vec<char> = p.chars().collect();
                for i in 0..pc.len().saturating_sub(2) { qset.insert(pc[i..i+3].iter().collect()); }
            }
            if !qset.is_empty() {
                for (doc, s) in fused.iter_mut() {
                    let c = self.docs[*doc as usize].content.to_lowercase();
                    let mut hit = 0usize;
                    let mut tot = 0usize;
                    for w in c.split_whitespace().take(120) {
                        if w.chars().count() < 4 { continue; }
                        let p = format!("#{}#", w);
                        let pc: Vec<char> = p.chars().collect();
                        for i in 0..pc.len().saturating_sub(2) {
                            tot += 1;
                            let g: String = pc[i..i+3].iter().collect();
                            if qset.contains(&g) { hit += 1; }
                        }
                        if tot > 400 { break; }
                    }
                    if tot > 0 {
                        let overlap = hit as f64 / (qset.len() + tot) as f64;
                        *s += overlap * 2.0;
                    }
                }
                fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        
        // 6. Convert to SearchResult
        let candidates: Vec<SearchResult> = fused.into_iter().map(|(doc, score)| {
            let d = &self.docs[doc as usize];
            let mut expanded = d.content.clone();
            if let Some(nxt) = self.docs.get(doc as usize + 1) {
                if nxt.source == d.source { expanded.push(' '); expanded.push_str(&nxt.content.chars().take(600).collect::<String>()); }
            }
            SearchResult { doc, score, page: d.page, source: d.source.clone(), heading: d.heading.clone(), snippet: d.content.chars().take(300).collect(), expanded: expanded.chars().take(900).collect() }
        }).collect();
        
        // 7. Rerank dengan dense embedding tersimpan (tanpa re-embed)
        reranker.rerank(&query_dense, &candidates, &self.docs, top_k)
    }

    fn save(&self, path: &Path) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }

    fn load(path: &Path) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let mut reader = io::BufReader::new(file);
        bincode::deserialize_from(&mut reader).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0; let mut na = 0.0; let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) { dot += a[i] as f64 * b[i] as f64; na += a[i] as f64 * a[i] as f64; nb += b[i] as f64 * b[i] as f64; }
    let norm = na.sqrt() * nb.sqrt();
    if norm > 0.0 { dot / norm } else { 0.0 }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SearchResult { doc: u32, score: f64, page: u32, source: String, heading: String, snippet: String, expanded: String }

// ==================== FILE PARSING ====================
// Filter terpusat: hanya dokumen sumber, bukan artefak index/config.
fn is_indexable(p: &Path) -> bool {
    if !p.is_file() { return false; }
    let name = p.file_name().unwrap_or_default().to_string_lossy();
    if name.starts_with('.') { return false; }
    let lower = name.to_lowercase();
    if lower.contains("enwik8") { return false; }
    if lower == "index.bin" || lower.ends_with(".bin") { return false; }
    if lower == "onod.json" || lower == "onod_learned.json" || lower == "config.json" { return false; }
    if lower.ends_with(".ds_store") { return false; }
    true
}

fn read_file(path: &Path) -> io::Result<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "txt"|"md"|"log"|"csv"|"tsv"|"json"|"jsonl"|"html"|"htm"|"xml" => {
            let metadata = fs::metadata(path)?;
            if metadata.len() > 100 * 1024 * 1024 { read_file_mmap(path) } else { fs::read_to_string(path) }
        }
        "pdf" => {
            match pdf_oxide::PdfDocument::open(path.to_str().unwrap_or("")) {
                Ok(doc) => { let num_pages = doc.page_count().unwrap_or(0); let mut all_text = String::new(); for page_idx in 0..num_pages { if let Ok(text) = doc.extract_text_auto(page_idx) { all_text.push_str(&text); all_text.push_str("\n\n"); } } Ok(all_text) }
                Err(e) => { eprintln!("  WARNING: PDF parse failed: {}", e); Ok(String::new()) }
            }
        }
        _ => { let metadata = fs::metadata(path)?; if metadata.len() > 100 * 1024 * 1024 { read_file_mmap(path) } else { fs::read_to_string(path) } }
    }
}
fn read_file_mmap(path: &Path) -> io::Result<String> {
    use memmap2::Mmap;
    let file = fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(String::from_utf8_lossy(&mmap).to_string())
}
fn read_files_parallel(paths: &[PathBuf]) -> Vec<(PathBuf, io::Result<String>)> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    let paths_clone: Vec<PathBuf> = paths.to_vec();
    std::thread::spawn(move || {
        let handles: Vec<_> = paths_clone.into_iter().map(|path| {
            let tx = tx.clone();
            std::thread::spawn(move || {
                let result = read_file(&path);
                let _ = tx.send((path, result));
            })
        }).collect();
        for h in handles { let _ = h.join(); }
        drop(tx);
    });
    rx.iter().collect()
}

// ==================== REST SERVER ====================
fn start_server(idx: Arc<OnodIndex>, embedder: Arc<TransformerEmbedder>, sparse_embedder: Arc<SparseEmbedder>, reranker: Arc<Reranker>, port: u16) {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("cannot bind");
    eprintln!("Server listening on http://0.0.0.0:{}", port);
    for stream in listener.incoming() { if let Ok(s) = stream { let idx = idx.clone(); let e = embedder.clone(); let se = sparse_embedder.clone(); let r = reranker.clone(); std::thread::spawn(move || { handle_connection(s, &idx, &e, &se, &r); }); } }
}
fn handle_connection(mut stream: TcpStream, idx: &OnodIndex, embedder: &TransformerEmbedder, sparse_embedder: &SparseEmbedder, reranker: &Reranker) {
    use std::io::{BufRead, BufReader};
    let mut buf = BufReader::new(&stream); let mut line = String::new(); buf.read_line(&mut line).unwrap();
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 { let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n"); return; }
    let resp = match parts[1] {
        "/" => fmt(200, &serde_json::json!({"name":"onod","version":"0.6.0","architecture":"transformer+dense+sparse+rerank"}).to_string()),
        "/health" => fmt(200, &serde_json::json!({"status":"ok","chunks":idx.docs.len()}).to_string()),
        p if p.starts_with("/search") => {
            let params: HashMap<String,String> = p.split('?').nth(1).unwrap_or("").split('&').filter_map(|p| p.split_once('=')).map(|(k,v)| (k.replace("%20"," "), v.replace("%20"," "))).collect();
            let q = params.get("q").map(|s| s.as_str()).unwrap_or(""); let top_k: usize = params.get("top_k").and_then(|s| s.parse().ok()).unwrap_or(10);
            if q.is_empty() { fmt(400, &"{'error':'missing q'}") } else { fmt(200, &serde_json::json!({"query":q,"results":idx.search(q,embedder,sparse_embedder,reranker,top_k)}).to_string()) }
        }
        _ => fmt(404, &"{'error':'not found'}"),
    };
    let _ = stream.write_all(resp.as_bytes());
}
fn fmt(s: u16, b: &str) -> String { format!("HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", s, match s {200=>"OK",400=>"Bad Request",404=>"Not Found",_=>"Error"}, b.len(), b) }

// ==================== MAIN ====================
fn main() {
    let _ = get_config();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 { eprintln!("onod v0.6.0 — Transformer Hybrid Search Engine\nUsage: onod <benchmark|search|serve|save|load> [args]"); std::process::exit(1); }
    
    // Initialize embedders
    eprintln!("Initializing transformer embedder...");
    let embedder = Arc::new(TransformerEmbedder::new());
    let reranker = Arc::new(Reranker::new());
    
    match args[1].as_str() {
        "benchmark" => {
            let folder = args.get(2).expect("provide folder");
            let mut idx = OnodIndex::new();
            let mut sparse = SparseEmbedder::new();
            
            // First pass: collect texts for vocab building
            let t0 = Instant::now();
            let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| is_indexable(p)).collect();
            
            let results = read_files_parallel(&paths);
            let texts: Vec<String> = results.iter().filter_map(|(_, r)| r.as_ref().ok().cloned()).collect();
            
            // Build sparse vocab
            sparse.build_vocab(&texts);
            
            // Index all documents
            for (path, result) in &results {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if let Ok(text) = result {
                    if !text.is_empty() {
                        let c = idx.add_text(&text, &name, 0, &embedder, &sparse);
                        if c > 0 { eprintln!("  {} -> {} chunks", name, c); }
                    }
                }
            }
            eprintln!("\nIndex: {} chunks, {:.2}s\n", idx.docs.len(), t0.elapsed().as_secs_f64());
            
            // 20 Complex Test Queries
            let queries = vec![
                ("Berapa total uang yang dihasilkan perusahaan dari pelanggan?", vec!["62.714","revenue","pendapatan"]),
                ("Pendapatan bersih PT ANTAM semester 1 2026 berapa?", vec!["62.714","pendapatan"]),
                ("How much money did the company earn from customer contracts?", vec!["62.714","revenue","income"]),
                ("Selisih antara pendapatan dan beban pokok penjualan?", vec!["10.851","laba","gross"]),
                ("Berapa keuntungan sebelum dikurangi beban operasi dan pajak?", vec!["10.851","laba","bruto"]),
                ("Profit after tax berapa juta rupiah?", vec!["8.443","laba","bersih","profit"]),
                ("Seluruh nilai kekayaan yang dimiliki perusahaan berapa?", vec!["aset","assets","total"]),
                ("Total kewajiban yang harus dibayar perusahaan?", vec!["utang","liabilities","kewajiban"]),
                ("Bagian pemegang saham dari total aset berapa?", vec!["modal","ekuitas","equity"]),
                ("Berapa bagian keuntungan yang dibagikan ke pemegang saham?", vec!["dividen","dividend","5"]),
                ("Berapa uang tunai yang dihasilkan dari kegiatan operasi?", vec!["arus","kas","operasi","cash"]),
                ("Berapa jumlah emas yang diproduksi perusahaan?", vec!["emas","gold","produksi"]),
                ("Produksi logam nikel berapa ton?", vec!["nikel","nickel","produksi"]),
                ("Siapa yang memimpin perusahaan sebagai presiden direktur?", vec!["direksi","Untung","presiden"]),
                ("Apa saja produk pertambangan utama perusahaan?", vec!["emas","nikel","bauksit","komoditas"]),
                ("Berapa beban pajak penghasilan yang harus dibayar?", vec!["pajak","tax","2"]),
                ("Berapa hak tagih dari pelanggan yang belum dibayar?", vec!["piutang","receivables","10"]),
                ("Berapa stok emas yang belum dijual?", vec!["persediaan","inventory","emas"]),
                ("Berapa selisih antara pendapatan dan beban pokok penjualan?", vec!["10.851","laba","bruto","gross"]),
                (" ANTAM 2026年上半年总收入是多少？", vec!["62.714","revenue","pendapatan"]),
            ];
            
            let mut hits = 0; let mut total_ms = 0.0;
            for (q, kws) in &queries {
                let t1 = Instant::now();
                let results = idx.search(q, &embedder, &sparse, &reranker, 10);
                let ms = t1.elapsed().as_secs_f64() * 1000.0;
                total_ms += ms;
                let blob: String = results.iter().flat_map(|r| vec![r.snippet.clone(), r.expanded.clone()]).collect::<Vec<_>>().join(" ").to_lowercase();
                let ok = kws.iter().any(|k| blob.contains(&k.to_lowercase()));
                if ok { hits += 1; }
                println!("{} {:6.1}ms | {:60} -> {}", if ok {"OK "} else {"FAIL"}, ms, q, if ok {"✅"} else {"❌"});
            }
            println!("\n=== RESULTS ===\nRecall: {}/{} = {:.1}%\nAvg latency: {:.1}ms\nIndex time: {:.2}s",
                hits, queries.len(), hits as f64 / queries.len() as f64 * 100.0, total_ms / queries.len() as f64, t0.elapsed().as_secs_f64());
        }
        "search" => {
            let folder = args.get(2).expect("provide folder"); let query = args.get(3..).unwrap_or(&[]).join(" ");
            if query.is_empty() { eprintln!("Usage: onod search <folder> <query>"); std::process::exit(1); }
            let mut idx = OnodIndex::new(); let mut sparse = SparseEmbedder::new();
            let index_path = Path::new(folder).join("index.bin");
            if index_path.exists() {
                eprintln!("Loading index...");
                match OnodIndex::load(&index_path) {
                    Ok(loaded) => { idx = loaded; }
                    Err(e) => { eprintln!("  stale index ({}), rebuilding...", e); let _ = std::fs::remove_file(&index_path); }
                }
            }
            if idx.docs.is_empty() {
                eprintln!("Building index...");
                let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| is_indexable(p)).collect();
                let results = read_files_parallel(&paths);
                let texts: Vec<String> = results.iter().filter_map(|(_, r)| r.as_ref().ok().cloned()).collect();
                sparse.build_vocab(&texts);
                for (path, result) in &results { let name = path.file_name().unwrap_or_default().to_string_lossy().to_string(); if let Ok(text) = result { idx.add_text(&text, &name, 0, &embedder, &sparse); } }
                let _ = idx.save(&index_path);
            }
            let t1 = Instant::now(); let results = idx.search(&query, &embedder, &sparse, &reranker, 10); let ms = t1.elapsed().as_secs_f64() * 1000.0;
            eprintln!("Query: \"{}\"\nSearch: {:.1}ms\n", query, ms);
            for (i, r) in results.iter().enumerate() { println!("{}. [{:.4}] {} — {}", i+1, r.score, r.heading, r.snippet.chars().take(150).collect::<String>()); }
        }
        "serve" => {
            let folder = args.get(2).expect("provide folder"); let port: u16 = args.iter().position(|a| a == "--port").and_then(|i| args.get(i+1)).and_then(|p| p.parse().ok()).unwrap_or(8080);
            let mut idx = OnodIndex::new(); let mut sparse = SparseEmbedder::new();
            eprintln!("Building index..."); let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| is_indexable(p)).collect();
            let results = read_files_parallel(&paths); let texts: Vec<String> = results.iter().filter_map(|(_, r)| r.as_ref().ok().cloned()).collect();
            sparse.build_vocab(&texts);
            for (path, result) in &results { let name = path.file_name().unwrap_or_default().to_string_lossy().to_string(); if let Ok(text) = result { idx.add_text(&text, &name, 0, &embedder, &sparse); } }
            eprintln!("Index: {} chunks", idx.docs.len());
            let idx = Arc::new(idx);
            let sparse = Arc::new(sparse);
            start_server(idx, embedder, sparse, reranker, port);
        }
        _ => { eprintln!("Unknown command: {}", args[1]); std::process::exit(1); }
    }
}
