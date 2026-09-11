use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use unicode_normalization::UnicodeNormalization;

// ==================== CONFIG ====================
const CHUNK_CHARS: usize = 600;
const CHUNK_OVERLAP: usize = 140;
const MIN_CHUNK: usize = 100;
const HARD_SPLIT: usize = 900;
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
const W_WORD: f64 = 0.72;
const W_TRI: f64 = 0.28;
const RRF_K: f64 = 60.0;

const STOPWORDS: &[&str] = &[
    "berapa","siapa","apa","bagaimana","kapan","mengapa","mana",
    "tolong","jelaskan","sebutkan","tunjukkan","berikan",
    "what","how","who","whom","whose","which","when","where","why",
    "is","are","do","does","did","the","a","an",
    "ne","kadar","kim","hangi","nedir","nasil","kac",
    "quoi","quel","quelle","est","sont","qing","shenme",
];

// ==================== NORMALIZATION ====================
fn is_ar_diacritic(c: char) -> bool {
    matches!(c, '\u{64B}'..='\u{652}' | '\u{670}')
}

fn normalize(raw: &str) -> String {
    let nfkc: String = raw.nfkc().collect();
    let mut out = String::with_capacity(nfkc.len());
    let mut prev_space = true;
    for c in nfkc.chars() {
        if is_ar_diacritic(c) { continue; }
        if c.is_whitespace() {
            if !prev_space { out.push(' '); prev_space = true; }
        } else { out.push(c); prev_space = false; }
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
    while i + 4 <= b.len() && b[i] == b'.' && b[i+1].is_ascii_digit() && b[i+2].is_ascii_digit() && b[i+3].is_ascii_digit() {
        i += 4; groups += 1;
    }
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
    let low = norm.to_lowercase();
    let words = tokenize_words_lower(&low);
    let base: Vec<String> = words.iter().filter(|w| !STOPWORDS.contains(&w.as_str())).cloned().collect();
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
    if start < ch.len() {
        let s: String = ch[start..].iter().collect();
        if !s.trim().is_empty() { parts.push(s.trim().to_string()); }
    }
    let mut out = Vec::with_capacity(parts.len());
    for s in parts {
        let mut rest = s.as_str();
        while rest.len() > HARD_SPLIT {
            let mut cut = HARD_SPLIT.min(rest.len());
            // Find a safe char boundary
            while cut > 0 && !rest.is_char_boundary(cut) { cut -= 1; }
            // Try to cut at a space
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
    let sentences = split_sentences(text);
    let mut chunks = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut buf_len = 0usize;
    let mut heading = String::new();
    for s in sentences {
        if is_heading(&s) {
            heading = s.trim_start_matches(['#',' ']).to_string();
            if heading.len()>120{heading.truncate(120);}
        }
        if buf_len + s.len() + 1 > CHUNK_CHARS && !buf.is_empty() {
            let content = buf.join(" ");
            if content.len() >= MIN_CHUNK { chunks.push((content, heading.clone())); }
            let mut tail = Vec::new(); let mut tl = 0;
            for item in buf.iter().rev() { tail.insert(0,item.clone()); tl+=item.len()+1; if tl>=CHUNK_OVERLAP||tail.len()>=2{break;} }
            buf = tail; buf_len = tl;
        }
        buf_len += s.len() + 1;
        buf.push(s);
    }
    if !buf.is_empty() {
        let content = buf.join(" ");
        if content.len() >= MIN_CHUNK { chunks.push((content, heading)); }
    }
    chunks
}

// ==================== INDEX ====================
struct Doc { content: String, source: String, page: u32, heading: String }

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
}

impl OnodIndex {
    fn new() -> Self {
        Self { docs:Vec::new(), w_post:HashMap::new(), t_post:HashMap::new(), w_len:Vec::new(), t_len:Vec::new(), w_idf:HashMap::new(), t_idf:HashMap::new(), w_avg:0.0, t_avg:0.0 }
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
        if norm.len() < MIN_CHUNK { return 0; }
        let chunks = chunk_text(&norm);
        let analyzed: Vec<_> = chunks.into_par_iter().map(|(c,h)|{
            let (lex,tri) = analyze_text(&c);
            (c,h,lex,tri)
        }).collect();
        let mut n = 0u32;
        for (content,heading,lex,tri) in analyzed {
            if lex.is_empty() && tri.is_empty(){continue;}
            let id = self.docs.len() as u32;
            self.docs.push(Doc{content,source:source.to_string(),page,heading});
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
        let mut qtf: HashMap<&str,u32> = HashMap::new();
        for t in qtokens{*qtf.entry(t.as_str()).or_insert(0)+=1;}
        let mut acc: HashMap<u32,f64> = HashMap::new();
        let mut matched: HashMap<u32,u32> = HashMap::new();
        for (t,qf) in &qtf {
            let plist = match post.get(*t){Some(v)=>v,None=>continue,};
            let idfv = match idf.get(*t){Some(v)=>*v,None=>continue,};
            if idfv<=0.0{continue;}
            for (doc,tf) in plist {
                let dl = lens.get(*doc as usize).copied().unwrap_or(avg as usize) as f64;
                let norm = 1.0 - BM25_B + BM25_B * dl / avg.max(1.0);
                let denom = *tf as f64 + BM25_K1 * norm;
                let s = idfv * (*tf as f64 * (BM25_K1+1.0) / denom);
                *acc.entry(*doc).or_insert(0.0) += s * (1.0+0.1*(*qf as f64-1.0));
                *matched.entry(*doc).or_insert(0) += 1;
            }
        }
        if acc.is_empty(){return Vec::new();}
        let nq = qtf.len().max(1) as f64;
        let mut scored: Vec<(u32,f64)> = acc.into_iter().map(|(doc,s)|{
            let m=matched.get(&doc).copied().unwrap_or(0) as f64;
            let cov=m/nq;
            let bonus=if (m-nq).abs()<1e-9{0.5}else{0.0};
            (doc,s*(0.4+0.6*cov+bonus))
        }).collect();
        let k = top_k.min(scored.len());
        if k == 0 { return scored; }
        scored.select_nth_unstable_by(k-1,|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        if self.docs.is_empty()||query.trim().is_empty(){return Vec::new();}
        let norm = normalize(query);
        let low = norm.to_lowercase();
        let words = tokenize_words_lower(&low);
        let base: Vec<String> = words.iter().filter(|w|!STOPWORDS.contains(&w.as_str())).cloned().collect();
        let mut lex = base.clone();
        for p in base.windows(2){if p[0].len()>2&&p[1].len()>2{lex.push(format!("{}_{}",p[0],p[1]));}}
        let mut tri = Vec::new();
        trigrams_of(&words,&mut tri);

        let wr = Self::bm25(&lex,&self.w_post,&self.w_idf,&self.w_len,self.w_avg,200);
        let tr = Self::bm25(&tri,&self.t_post,&self.t_idf,&self.t_len,self.t_avg,200);

        let mut acc: HashMap<u32,f64> = HashMap::new();
        for (rank,(doc,_)) in wr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=W_WORD/(RRF_K+rank as f64+1.0);}
        for (rank,(doc,_)) in tr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=W_TRI/(RRF_K+rank as f64+1.0);}

        let qnums: HashSet<&str> = lex.iter().filter(|t|is_thousands(t)).map(|s|s.as_str()).collect();
        let mut fused: Vec<(u32,f64)> = acc.into_iter().collect();
        if !qnums.is_empty(){
            for(doc,s) in fused.iter_mut(){
                let c=&self.docs[*doc as usize].content;
                for qn in &qnums{if c.contains(*qn){*s+=0.6;break;}}
            }
        }
        fused.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);

        fused.into_iter().map(|(doc,score)|{
            let d = &self.docs[doc as usize];
            let mut expanded = d.content.clone();
            if let Some(nxt) = self.docs.get(doc as usize + 1) {
                if nxt.source == d.source {
                    expanded.push(' ');
                    let trunc: String = nxt.content.chars().take(600).collect();
                    expanded.push_str(&trunc);
                }
            }
            SearchResult {
                doc, score, page: d.page,
                source: d.source.clone(), heading: d.heading.clone(),
                snippet: d.content.chars().take(300).collect(),
                expanded: expanded.chars().take(900).collect(),
            }
        }).collect()
    }
}

#[derive(serde::Serialize)]
struct SearchResult {
    doc: u32, score: f64, page: u32,
    source: String, heading: String,
    snippet: String, expanded: String,
}

// ==================== FILE PARSING ====================
fn read_file(path: &Path) -> io::Result<String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "txt"|"md"|"log"|"csv"|"tsv"|"json"|"jsonl"|"html"|"htm"|"xml" => {
            fs::read_to_string(path)
        }
        "pdf" => {
            // Use pymupdf via subprocess
            let output = std::process::Command::new("python3")
                .args(["-c", &format!(
                    "import pymupdf; doc=pymupdf.open('{}'); print('\\n\\n'.join(p.get_text() for p in doc))",
                    path.display()
                )])
                .output()?;
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
        _ => {
            fs::read_to_string(path)
        }
    }
}

// ==================== MAIN ====================
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: onod <command> [args]");
        eprintln!("Commands:");
        eprintln!("  index <folder>        Index all files in folder");
        eprintln!("  search <index> <query> Search an indexed folder");
        eprintln!("  benchmark <folder>     Run accuracy benchmark");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "index" => {
            let folder = args.get(2).expect("provide folder");
            let mut idx = OnodIndex::new();
            let t0 = Instant::now();

            let paths: Vec<PathBuf> = fs::read_dir(folder)
                .expect("cannot read dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.'))
                .collect();

            for (i, path) in paths.iter().enumerate() {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                eprint!("  [{}/{}] {} ... ", i+1, paths.len(), name);
                match read_file(path) {
                    Ok(text) => {
                        let chunks = idx.add_text(&text, &name, 0);
                        eprintln!("{} chunks", chunks);
                    }
                    Err(e) => eprintln!("ERROR: {}", e),
                }
            }
            idx.finalize();
            let elapsed = t0.elapsed();
            eprintln!("\nIndexed {} files, {} chunks, {} word terms, {} tri terms in {:.2}s",
                paths.len(), idx.docs.len(), idx.w_post.len(), idx.t_post.len(), elapsed.as_secs_f64());
        }
        "search" => {
            let index_dir = args.get(2).expect("provide index dir");
            let query = args.get(3).expect("provide query");
            // For now just read from folder directly
            eprintln!("Search not implemented yet - use benchmark mode");
        }
        "benchmark" => {
            let folder = args.get(2).expect("provide folder with documents");
            let mut idx = OnodIndex::new();
            let t0 = Instant::now();

            let paths: Vec<PathBuf> = fs::read_dir(folder)
                .expect("cannot read dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    if !p.is_file() { return false; }
                    let name = p.file_name().unwrap_or_default().to_string_lossy();
                    if name.starts_with('.') || name == "enwik8" || name.ends_with(".DS_Store") { return false; }
                    true
                })
                .collect();

            for (i, path) in paths.iter().enumerate() {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                eprint!("  [{}/{}] {} ... ", i+1, paths.len(), name);
                match read_file(path) {
                    Ok(text) => {
                        let chunks = idx.add_text(&text, &name, 0);
                        eprintln!("{} chunks", chunks);
                    }
                    Err(e) => eprintln!("ERROR: {}", e),
                }
            }
            idx.finalize();
            let index_time = t0.elapsed();
            eprintln!("\nIndex: {} chunks, {:.2}s\n", idx.docs.len(), index_time.as_secs_f64());

            // Benchmark queries
            let queries = vec![
                ("What is the total revenue?", vec!["revenue", "pendapatan"]),
                ("Siapa saja direksi?", vec!["direksi", "director"]),
                ("Berapa laba bruto?", vec!["laba", "gross", "profit"]),
                ("What are the main commodities?", vec!["gold", "nickel", "emas"]),
                ("Net profit margin?", vec!["profit", "margin"]),
            ];

            let mut hits = 0;
            let mut total_ms = 0.0;
            for (q, kws) in &queries {
                let t1 = Instant::now();
                let results = idx.search(q, 5);
                let ms = t1.elapsed().as_secs_f64() * 1000.0;
                total_ms += ms;

                let blob: String = results.iter()
                    .flat_map(|r| vec![r.snippet.clone(), r.expanded.clone()])
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase();

                let ok = kws.iter().any(|k| blob.contains(&k.to_lowercase()));
                if ok { hits += 1; }
                println!("{} {:6.1}ms | {:50} -> {}",
                    if ok {"OK "} else {"FAIL"},
                    ms, q,
                    results.first().map(|r| r.snippet[..60.min(r.snippet.len())].to_string()).unwrap_or_default()
                );
            }

            let recall = hits as f64 / queries.len() as f64;
            let avg_ms = total_ms / queries.len() as f64;
            println!("\n=== RESULTS ===");
            println!("Recall: {}/{} = {:.1}%", hits, queries.len(), recall * 100.0);
            println!("Avg latency: {:.1}ms", avg_ms);
            println!("Index time: {:.2}s", index_time.as_secs_f64());
            if recall >= 0.98 && avg_ms < 100.0 {
                println!("PASS - accuracy >98%, latency <100ms");
            } else {
                println!("FAIL - need tuning");
            }
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}
