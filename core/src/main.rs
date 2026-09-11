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

// ==================== CONFIG ====================
#[derive(serde::Deserialize, serde::Serialize, Clone)]
struct Config {
    chunk_chars: usize,
    chunk_overlap: usize,
    min_chunk: usize,
    hard_split: usize,
    bm25_k1: f64,
    bm25_b: f64,
    w_word: f64,
    w_tri: f64,
    w_dense: f64,
    rrf_k: f64,
    financial_num_threshold: usize,
    financial_term_threshold: usize,
    num_boost: f64,
    financial_boost: f64,
    embedding_dim: usize,
    dense_weight: f64,
    sparse_weight: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_chars: 600,
            chunk_overlap: 140,
            min_chunk: 100,
            hard_split: 900,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            w_word: 0.72,
            w_tri: 0.28,
            w_dense: 0.50,
            rrf_k: 60.0,
            financial_num_threshold: 50,
            financial_term_threshold: 20,
            num_boost: 0.3,
            financial_boost: 0.4,
            embedding_dim: 384,
            dense_weight: 0.6,
            sparse_weight: 0.4,
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

// ==================== DENSE EMBEDDING INDEX ====================
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct DenseIndex {
    embeddings: Vec<Vec<f32>>,
    doc_ids: Vec<u32>,
    dim: usize,
}

impl DenseIndex {
    fn new(dim: usize) -> Self { Self { embeddings: Vec::new(), doc_ids: Vec::new(), dim } }
    
    fn add(&mut self, doc_id: u32, embedding: Vec<f32>) {
        self.embeddings.push(embedding);
        self.doc_ids.push(doc_id);
    }
    
    fn search(&self, query: &[f32], top_k: usize) -> Vec<(u32, f64)> {
        if self.embeddings.is_empty() { return Vec::new(); }
        let mut scores: Vec<(u32, f64)> = self.embeddings.iter().zip(self.doc_ids.iter())
            .map(|(emb, &doc_id)| (doc_id, cosine_similarity(query, emb)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }
    
    fn save(&self, path: &Path) -> io::Result<()> {
        let data = serde_json::json!({
            "dim": self.dim,
            "embeddings": self.embeddings,
            "doc_ids": self.doc_ids,
        });
        fs::write(path, serde_json::to_string(&data)?)
    }
    
    fn load(path: &Path) -> io::Result<Self> {
        let data: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
        let dim = data["dim"].as_u64().unwrap_or(384) as usize;
        let embeddings = data["embeddings"].as_array().unwrap().iter()
            .map(|e| e.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect();
        let doc_ids = data["doc_ids"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as u32).collect();
        Ok(Self { embeddings, doc_ids, dim })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0; let mut na = 0.0; let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) { dot += a[i] as f64 * b[i] as f64; na += a[i] as f64 * a[i] as f64; nb += b[i] as f64 * b[i] as f64; }
    let norm = na.sqrt() * nb.sqrt();
    if norm > 0.0 { dot / norm } else { 0.0 }
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
#[inline]
fn is_thousands(tok: &str) -> bool {
    let b = tok.as_bytes();
    let mut i = 0; let mut head = 0;
    while i < b.len() && b[i].is_ascii_digit() && head < 3 { i += 1; head += 1; }
    if head == 0 || head > 3 { return false; }
    let mut groups = 0;
    while i + 4 <= b.len() && b[i] == b'.' && b[i+1].is_ascii_digit() && b[i+2].is_ascii_digit() && b[i+3].is_ascii_digit() { i += 4; groups += 1; }
    if groups == 0 { return false; }
    if i < b.len() && b[i] == b',' { i += 1; let s = i; while i < b.len() && b[i].is_ascii_digit() { i += 1; } if i == s { return false; } }
    i == b.len()
}

// ==================== TOKENIZER ====================
fn tokenize_words_lower(low: &str) -> Vec<String> {
    let mut words = Vec::new();
    let chars: Vec<char> = low.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            let mut j = i; let mut head = 0;
            while j < chars.len() && chars[j].is_ascii_digit() && head < 3 { j += 1; head += 1; }
            let mut k = j; let mut groups = 0;
            while k+4 <= chars.len() && chars[k]=='.' && chars[k+1].is_ascii_digit() && chars[k+2].is_ascii_digit() && chars[k+3].is_ascii_digit() { k+=4; groups+=1; }
            if groups > 0 {
                let mut kk = k;
                if kk < chars.len() && chars[kk] == ',' { kk+=1; let s=kk; while kk<chars.len() && chars[kk].is_ascii_digit(){kk+=1;} if kk>s{k=kk;} }
                if k>=chars.len() || !chars[k].is_alphanumeric() { words.push(chars[i..k].iter().collect()); i=k; continue; }
            }
            let mut j2 = i;
            while j2<chars.len() && chars[j2].is_alphanumeric(){j2+=1;}
            let t: String = chars[i..j2].iter().collect();
            if t.len() > 1 || t.chars().all(|x| x.is_alphanumeric()) { words.push(t); }
            i = j2;
        } else if c.is_alphanumeric() {
            let mut j = i;
            while j<chars.len() && chars[j].is_alphanumeric(){j+=1;}
            let t: String = chars[i..j].iter().collect();
            if t.len() > 1 || t.chars().all(|x| x.is_alphanumeric()) { words.push(t); }
            i = j;
        } else { i += 1; }
    }
    words
}

fn trigrams_of(words: &[String], out: &mut Vec<String>) {
    for w in words {
        if w.chars().count() < 4 || is_thousands(w) { continue; }
        let p = format!("#{}#", w);
        let pc: Vec<char> = p.chars().collect();
        for i in 0..pc.len().saturating_sub(2) { out.push(pc[i..i+3].iter().collect()); }
    }
}

fn analyze_text(norm: &str) -> (Vec<String>, Vec<String>) {
    let sw: HashSet<&str> = ["the","a","an","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","must","can","could","of","in","to","for","with","on","at","from","by","about","as","into","through","during","before","after","above","below","between","out","off","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","just","because","but","and","or","if","while","although","though","since","until","unless","bagaimana","siapa","apa","berapa","kapan","dimana","mengapa","tolong","jelaskan","sebutkan","tunjukkan","berikan"].iter().cloned().collect();
    let low = norm.to_lowercase();
    let words = tokenize_words_lower(&low);
    let base: Vec<String> = words.iter().filter(|w| !sw.contains(w.as_str())).cloned().collect();
    let mut lex = base.clone();
    for p in base.windows(2) { if p[0].len()>2 && p[1].len()>2 { lex.push(format!("{}_{}",p[0],p[1])); } }
    let mut tri = Vec::new();
    trigrams_of(&words, &mut tri);
    (lex, tri)
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
        if is_heading(&s) { heading = s.trim_start_matches(['#',' ']).to_string(); if heading.len()>120{heading.truncate(120);} }
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
struct Doc { content: String, source: String, page: u32, heading: String, embedding: Option<Vec<f32>> }

#[derive(serde::Serialize, serde::Deserialize)]
struct OnodIndex {
    docs: Vec<Doc>,
    w_post: HashMap<String, Vec<(u32,u32)>>,
    t_post: HashMap<String, Vec<(u32,u32)>>,
    w_len: Vec<usize>,
    t_len: Vec<usize>,
    w_idf: HashMap<String, f64>,
    t_idf: HashMap<String, f64>,
    w_avg: f64,
    t_avg: f64,
    dense_index: DenseIndex,
}

impl OnodIndex {
    fn new() -> Self {
        let cfg = get_config();
        Self { docs:Vec::new(), w_post:HashMap::new(), t_post:HashMap::new(), w_len:Vec::new(), t_len:Vec::new(), w_idf:HashMap::new(), t_idf:HashMap::new(), w_avg:0.0, t_avg:0.0, dense_index: DenseIndex::new(cfg.embedding_dim) }
    }

    fn finalize(&mut self) {
        let n = self.docs.len();
        if n==0{return;}
        self.w_avg = self.w_len.iter().sum::<usize>() as f64 / n as f64;
        self.t_avg = self.t_len.iter().sum::<usize>() as f64 / n as f64;
        let mut wi = HashMap::with_capacity(self.w_post.len());
        for (t,v) in &self.w_post { let df=v.len() as f64; wi.insert(t.clone(), ((n as f64-df+0.5)/(df+0.5)+1.0).ln()); }
        let mut ti = HashMap::with_capacity(self.t_post.len());
        for (t,v) in &self.t_post { let df=v.len() as f64; ti.insert(t.clone(), ((n as f64-df+0.5)/(df+0.5)+1.0).ln()); }
        self.w_idf = wi; self.t_idf = ti;
    }

    fn add_text(&mut self, text: &str, source: &str, page: u32) -> u32 {
        let norm = normalize(text);
        let cfg = get_config();
        if norm.len() < cfg.min_chunk { return 0; }
        let chunks = chunk_text(&norm);
        let analyzed: Vec<_> = chunks.into_par_iter().map(|(c,h)|{ let (lex,tri) = analyze_text(&c); (c,h,lex,tri) }).collect();
        let mut n = 0u32;
        for (content,heading,lex,tri) in analyzed {
            if lex.is_empty() && tri.is_empty(){continue;}
            let id = self.docs.len() as u32;
            let embedding = simple_hash_embedding(&content, cfg.embedding_dim);
            self.docs.push(Doc{content,source:source.to_string(),page,heading,embedding:Some(embedding.clone())});
            self.dense_index.add(id, embedding);
            self.w_len.push(lex.len()); self.t_len.push(tri.len());
            let mut tf:HashMap<&str,u32> = HashMap::new();
            for t in &lex{*tf.entry(t.as_str()).or_insert(0)+=1;}
            for (t,f) in tf{self.w_post.entry(t.to_string()).or_default().push((id,f));}
            let mut tf2:HashMap<&str,u32> = HashMap::new();
            for t in &tri{*tf2.entry(t.as_str()).or_insert(0)+=1;}
            for (t,f) in tf2{self.t_post.entry(t.to_string()).or_default().push((id,f));}
            n+=1;
        }
        n
    }

    fn bm25(qtokens: &[String], post: &HashMap<String,Vec<(u32,u32)>>, idf: &HashMap<String,f64>, lens: &[usize], avg: f64, top_k: usize) -> Vec<(u32,f64)> {
        let cfg = get_config();
        let mut qtf: HashMap<&str,u32> = HashMap::new();
        for t in qtokens{*qtf.entry(t.as_str()).or_insert(0)+=1;}
        let mut acc: HashMap<u32,f64> = HashMap::new();
        let mut matched: HashMap<u32,u32> = HashMap::new();
        for (t,qf) in &qtf {
            let plist = match post.get(*t){Some(v)=>v,None=>continue,};
            let idfv = match idf.get(*t){Some(v)=>*v,None=>continue,};
            if idfv<=0.0{continue;}
            for chunk in plist.chunks(8) {
                let chunk_len = chunk.len();
                let mut batch_dl: [f64; 8] = [avg; 8];
                let mut batch_tf: [f64; 8] = [0.0; 8];
                let mut batch_doc: [u32; 8] = [0; 8];
                for (i, (doc_id, tf)) in chunk.iter().enumerate() { batch_doc[i] = *doc_id; batch_dl[i] = lens.get(*doc_id as usize).copied().unwrap_or(avg as usize) as f64; batch_tf[i] = *tf as f64; }
                let k1_plus_1 = cfg.bm25_k1 + 1.0; let one_minus_b = 1.0 - cfg.bm25_b; let inv_avg = 1.0 / avg.max(1.0);
                let mut batch_scores: [f64; 8] = [0.0; 8];
                for i in 0..chunk_len { let norm = one_minus_b + cfg.bm25_b * batch_dl[i] * inv_avg; let denom = batch_tf[i] + cfg.bm25_k1 * norm; batch_scores[i] = if denom > 0.0 { idfv * (batch_tf[i] * k1_plus_1 / denom) } else { 0.0 }; }
                let nq = qtf.len().max(1) as f64;
                for i in 0..chunk_len { let score = batch_scores[i]; if score > 0.0 { let m = matched.entry(batch_doc[i]).or_insert(0); *m += 1; let cov = *m as f64 / nq; let bonus = if (*m as f64 - nq).abs() < 1e-9 { 0.5 } else { 0.0 }; *acc.entry(batch_doc[i]).or_insert(0.0) += score * (0.4 + 0.6 * cov + bonus) * (1.0 + 0.1 * (*qf as f64 - 1.0)); } }
            }
        }
        if acc.is_empty(){return Vec::new();}
        let mut scored: Vec<(u32,f64)> = acc.into_iter().collect();
        let k = top_k.min(scored.len());
        if k == 0 { return scored; }
        scored.select_nth_unstable_by(k-1,|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn hybrid_search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        if self.docs.is_empty()||query.trim().is_empty(){return Vec::new();}
        let cfg = get_config();
        let sw: HashSet<&str> = ["the","a","an","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","must","can","could","of","in","to","for","with","on","at","from","by","about","as","into","through","during","before","after","above","below","between","out","off","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","just","because","but","and","or","if","while","although","though","since","until","unless","bagaimana","siapa","apa","berapa","kapan","dimana","mengapa","tolong","jelaskan","sebutkan","tunjukkan","berikan"].iter().cloned().collect();
        
        // 1. Sparse search (BM25 word + trigram)
        let norm = normalize(query);
        let low = norm.to_lowercase();
        let words = tokenize_words_lower(&low);
        let base: Vec<String> = words.iter().filter(|w|!sw.contains(w.as_str())).cloned().collect();
        let mut lex = base.clone();
        for p in base.windows(2){if p[0].len()>2&&p[1].len()>2{lex.push(format!("{}_{}",p[0],p[1]));}}
        let mut tri = Vec::new();
        trigrams_of(&words,&mut tri);
        let wr = Self::bm25(&lex,&self.w_post,&self.w_idf,&self.w_len,self.w_avg,200);
        let tr = Self::bm25(&tri,&self.t_post,&self.t_idf,&self.t_len,self.t_avg,200);
        
        // 2. Dense search (embedding similarity)
        let query_embedding = simple_hash_embedding(&query, cfg.embedding_dim);
        let dense_results = self.dense_index.search(&query_embedding, 200);
        
        // 3. Hybrid fusion (RRF)
        let mut acc: HashMap<u32,f64> = HashMap::new();
        for (rank,(doc,_)) in wr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=cfg.w_word/(cfg.rrf_k+rank as f64+1.0);}
        for (rank,(doc,_)) in tr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=cfg.w_tri/(cfg.rrf_k+rank as f64+1.0);}
        for (rank,(doc,_)) in dense_results.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=cfg.w_dense/(cfg.rrf_k+rank as f64+1.0);}
        
        // 4. Boosts
        let qnums: HashSet<&str> = lex.iter().filter(|t|is_thousands(t)).map(|s|s.as_str()).collect();
        let mut fused: Vec<(u32,f64)> = acc.into_iter().collect();
        if !qnums.is_empty(){for(doc,s) in fused.iter_mut(){let c=&self.docs[*doc as usize].content;for qn in &qnums{if c.contains(*qn){*s+=0.6;break;}}}}
        let fin_terms: HashSet<&str> = ["revenue","income","pendapatan","profit","laba","assets","aset","liabilitas","utang","equity","ekuitas","modal","dividen","dividend","arus","kas","cash","penjualan","sales","beban","cost","expense","tax","pajak","emas","gold","nikel","nickel","bauksit","bauxite"].iter().cloned().collect();
        for(doc,s) in fused.iter_mut(){let c = self.docs[*doc as usize].content.to_lowercase();let nc=c.matches(|c:char|c.is_ascii_digit()).count();if nc>cfg.financial_num_threshold{*s+=cfg.num_boost;}let hf=fin_terms.iter().any(|t|c.contains(t));if hf&&nc>cfg.financial_term_threshold{*s+=cfg.financial_boost;}}
        
        // 5. Rerank with dense similarity boost
        for(doc,s) in fused.iter_mut(){
            if let Some(emb) = &self.docs[*doc as usize].embedding {
                let dense_sim = cosine_similarity(&query_embedding, emb);
                *s += dense_sim * cfg.dense_weight;
            }
        }
        
        fused.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused.into_iter().map(|(doc,score)|{
            let d = &self.docs[doc as usize];
            let mut expanded_content = d.content.clone();
            if let Some(nxt) = self.docs.get(doc as usize + 1) { if nxt.source == d.source { expanded_content.push(' '); expanded_content.push_str(&nxt.content.chars().take(600).collect::<String>()); } }
            SearchResult { doc, score, page: d.page, source: d.source.clone(), heading: d.heading.clone(), snippet: d.content.chars().take(300).collect(), expanded: expanded_content.chars().take(900).collect() }
        }).collect()
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

// Simple hash-based embedding (placeholder for ONNX model)
fn simple_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dim];
    let words: Vec<&str> = text.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        let hash = word.len() as f32 * 0.1 + i as f32 * 0.01;
        let idx = (hash * dim as f32) as usize % dim;
        embedding[idx] += 1.0;
    }
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { for x in &mut embedding { *x /= norm; } }
    embedding
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SearchResult { doc: u32, score: f64, page: u32, source: String, heading: String, snippet: String, expanded: String }

// ==================== FILE PARSING ====================
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
fn start_server(idx: Arc<OnodIndex>, port: u16) {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("cannot bind");
    eprintln!("Server listening on http://0.0.0.0:{}", port);
    for stream in listener.incoming() { if let Ok(s) = stream { let idx = idx.clone(); std::thread::spawn(move || { handle_connection(s, &idx); }); } }
}
fn handle_connection(mut stream: TcpStream, idx: &OnodIndex) {
    use std::io::{BufRead, BufReader};
    let mut buf = BufReader::new(&stream); let mut line = String::new(); buf.read_line(&mut line).unwrap();
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 { let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n"); return; }
    let resp = match parts[1] {
        "/" => fmt(200, &serde_json::json!({"name":"onod","version":"0.5.0"}).to_string()),
        "/health" => fmt(200, &serde_json::json!({"status":"ok","chunks":idx.docs.len()}).to_string()),
        p if p.starts_with("/search") => {
            let params: HashMap<String,String> = p.split('?').nth(1).unwrap_or("").split('&').filter_map(|p| p.split_once('=')).map(|(k,v)| (k.replace("%20"," "), v.replace("%20"," "))).collect();
            let q = params.get("q").map(|s| s.as_str()).unwrap_or(""); let top_k: usize = params.get("top_k").and_then(|s| s.parse().ok()).unwrap_or(10);
            if q.is_empty() { fmt(400, &"{'error':'missing q'}") } else { fmt(200, &serde_json::json!({"query":q,"results":idx.hybrid_search(q,top_k)}).to_string()) }
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
    if args.len() < 2 { eprintln!("onod v0.5.0 — Hybrid Dense+Sparse Search Engine\nUsage: onod <benchmark|search|serve|save|load> [args]"); std::process::exit(1); }
    match args[1].as_str() {
        "benchmark" => {
            let folder = args.get(2).expect("provide folder"); let mut idx = OnodIndex::new(); let t0 = Instant::now();
            let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.') && !p.file_name().unwrap_or_default().to_string_lossy().contains("enwik8")).collect();
            for (path, result) in read_files_parallel(&paths) { let name = path.file_name().unwrap_or_default().to_string_lossy().to_string(); if let Ok(text) = result { if !text.is_empty() { let c = idx.add_text(&text, &name, 0); if c > 0 { eprintln!("  {} -> {} chunks", name, c); } } } }
            idx.finalize(); eprintln!("\nIndex: {} chunks, {:.2}s\n", idx.docs.len(), t0.elapsed().as_secs_f64());
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
                let t1 = Instant::now(); let results = idx.hybrid_search(q, 10); let ms = t1.elapsed().as_secs_f64() * 1000.0; total_ms += ms;
                let blob: String = results.iter().flat_map(|r| vec![r.snippet.clone(), r.expanded.clone()]).collect::<Vec<_>>().join(" ").to_lowercase();
                let ok = kws.iter().any(|k| blob.contains(&k.to_lowercase())); if ok { hits += 1; }
                println!("{} {:6.1}ms | {:60} -> {}", if ok {"OK "} else {"FAIL"}, ms, q, if ok {"✅"} else {"❌"});
            }
            println!("\n=== RESULTS ===\nRecall: {}/{} = {:.1}%\nAvg latency: {:.1}ms\nIndex time: {:.2}s", hits, queries.len(), hits as f64 / queries.len() as f64 * 100.0, total_ms / queries.len() as f64, t0.elapsed().as_secs_f64());
        }
        "search" => {
            let folder = args.get(2).expect("provide folder"); let query = args.get(3..).unwrap_or(&[]).join(" ");
            if query.is_empty() { eprintln!("Usage: onod search <folder> <query>"); std::process::exit(1); }
            let mut idx = OnodIndex::new(); let index_path = Path::new(folder).join("index.bin");
            if index_path.exists() { eprintln!("Loading index..."); idx = OnodIndex::load(&index_path).expect("failed to load"); }
            else { eprintln!("Building index..."); let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.') && !p.file_name().unwrap_or_default().to_string_lossy().contains("enwik8")).collect(); for (path, result) in read_files_parallel(&paths) { let name = path.file_name().unwrap_or_default().to_string_lossy().to_string(); if let Ok(text) = result { idx.add_text(&text, &name, 0); } } idx.finalize(); let _ = idx.save(&index_path); }
            let t1 = Instant::now(); let results = idx.hybrid_search(&query, 10); let ms = t1.elapsed().as_secs_f64() * 1000.0;
            eprintln!("Query: \"{}\"\nSearch: {:.1}ms\n", query, ms);
            for (i, r) in results.iter().enumerate() { println!("{}. [{:.4}] {} — {}", i+1, r.score, r.heading, r.snippet.chars().take(150).collect::<String>()); }
        }
        "serve" => {
            let folder = args.get(2).expect("provide folder"); let port: u16 = args.iter().position(|a| a == "--port").and_then(|i| args.get(i+1)).and_then(|p| p.parse().ok()).unwrap_or(8080);
            let mut idx = OnodIndex::new(); eprintln!("Building index..."); let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.') && !p.file_name().unwrap_or_default().to_string_lossy().contains("enwik8")).collect();
            for (path, result) in read_files_parallel(&paths) { let name = path.file_name().unwrap_or_default().to_string_lossy().to_string(); if let Ok(text) = result { idx.add_text(&text, &name, 0); } } idx.finalize(); eprintln!("Index: {} chunks", idx.docs.len()); start_server(Arc::new(idx), port);
        }
        "save" => {
            let folder = args.get(2).expect("provide folder"); let save_path = args.get(3).expect("provide output path"); let mut idx = OnodIndex::new(); let t0 = Instant::now();
            let paths: Vec<PathBuf> = fs::read_dir(folder).expect("cannot read dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.') && !p.file_name().unwrap_or_default().to_string_lossy().contains("enwik8")).collect();
            for (path, result) in read_files_parallel(&paths) { let name = path.file_name().unwrap_or_default().to_string_lossy().to_string(); if let Ok(text) = result { idx.add_text(&text, &name, 0); } } idx.finalize(); let p = Path::new(save_path);
            if p.extension().map(|e| e.to_str() == Some("json")).unwrap_or(false) { let _ = fs::write(p, serde_json::to_string_pretty(&idx).unwrap()); } else { idx.save(p).expect("failed"); }
            eprintln!("Saved: {} chunks -> {} ({:.2}s)", idx.docs.len(), save_path, t0.elapsed().as_secs_f64());
        }
        "load" => {
            let load_path = args.get(2).expect("provide index file"); let idx = OnodIndex::load(Path::new(load_path)).expect("failed to load"); eprintln!("Loaded: {} chunks", idx.docs.len());
            loop { print!("> "); io::stdout().flush().unwrap(); let mut input = String::new(); if io::stdin().read_line(&mut input).is_err() { break; } let input = input.trim(); if input=="quit"||input=="exit" { break; } if input.is_empty() { continue; }
                for (i, r) in idx.hybrid_search(input, 10).iter().enumerate() { println!("  {}. [{:.4}] {}", i+1, r.score, r.snippet.chars().take(120).collect::<String>()); }
            }
        }
        _ => { eprintln!("Unknown command: {}", args[1]); std::process::exit(1); }
    }
}
