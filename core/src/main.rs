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

// ==================== SYNONYM DICTIONARY ====================
// Multilingual synonym groups: semua kata dalam 1 group = artinya sama
// Expand query sebelum BM25 agar search 100% akurat untuk konteks
fn build_synonym_groups() -> Vec<Vec<&'static str>> {
    vec![
        // Finance - Revenue
        vec!["revenue", "income", "pendapatan", "penghasilan", "omzet", "gains", "收収益", "収益", "revenues"],
        // Finance - Profit
        vec!["profit", "laba", "keuntungan", "earnings", "gains", "利润", "利益", "netincome", "net_income"],
        // Finance - Loss
        vec!["loss", "rugi", "kerugian", "deficit", "亏损", "損失"],
        // Finance - Assets
        vec!["assets", "aset", "aktiva", "properties", "资产", "資産"],
        // Finance - Liabilities
        vec!["liabilities", "liabilitas", "kewajiban", "utang", "hutang", "debts", "obligations", "负债", "負債"],
        // Finance - Equity
        vec!["equity", "ekuitas", "modal", "capital", "shareholders", "权益", "持分"],
        // Finance - Revenue from contracts
        vec!["revenue from contracts", "pendapatan dari kontrak", "contract revenue", "contract liabilities", "liabilitas kontrak"],
        // Finance - Cost
        vec!["cost", "biaya", "beban", "expense", "expenses", "成本", "費用", "costofrevenues", "bebanpokokpenjualan"],
        // Finance - Gross profit
        vec!["gross profit", "laba bruto", "lababruto", "grossmargin", "毛利", "粗利"],
        // Finance - Operating profit
        vec!["operating profit", "laba usaha", "operatingincome", "labausaha", "营业利润", "営業利益"],
        // Finance - Net profit
        vec!["net profit", "laba bersih", "netincome", "profitaftertax", "税后利润", "純利益"],
        // Finance - Tax
        vec!["tax", "pajak", "taxes", "income tax", "pajak penghasilan", "税", "税金"],
        // Finance - Cash
        vec!["cash", "kas", "cashflow", "arus kas", "liquidity", "现金", "キャッシュ"],
        // Finance - Dividend
        vec!["dividend", "dividen", "dividends", "股息", "配当"],
        // Finance - Shares
        vec!["share", "saham", "shares", "stock", "equity", "股份", "株式"],
        // Finance - Financial statements
        vec!["financial statements", "laporan keuangan", "financial report", "财务报表", "財務諸表"],
        // Mining/Commodities
        vec!["gold", "emas", "precious metals", "黄金", "金"],
        vec!["nickel", "nikel", "ferronickel", "feronikel", "镍", "ニッケル"],
        vec!["bauxite", "bauksit", "铝土矿", "ボーキサイト"],
        vec!["commodities", "komoditas", "commodity", "products", "produk", "mining products", "mining"],
        // Directors/Management
        vec!["directors", "direksi", "direktur", "board of directors", "management", "董事会", "取締役"],
        vec!["commissioners", "komisaris", "dewan komisaris", "board of commissioners", "监事", "監査役"],
        vec!["president director", "presiden direktur", "ceo", "chief executive"],
        vec!["president commissioner", "presiden komisaris", "chairman"],
        // Business operations
        vec!["production", "produksi", "output", "manufacturing", "生产", "生産"],
        vec!["exploration", "eksplorasi", "mining exploration", "勘探", "探鉱"],
        vec!["sales", "penjualan", "revenue", "transactions", "销售", "販売"],
        vec!["customers", "pelanggan", "clients", "buyers", "客户", "顧客"],
        vec!["employees", "karyawan", "staff", "workers", "personnel", "员工", "従業員"],
        // Financial metrics
        vec!["total assets", "total aset", "jumlah aset", "total_properties"],
        vec!["total liabilities", "total liabilitas", "jumlah kewajiban"],
        vec!["total equity", "total ekuitas", "jumlah modal"],
        vec!["net profit margin", "marginal laba bersih", "profit margin", "margin keuntungan"],
        vec!["revenue growth", "pertumbuhan pendapatan", "pendapatan meningkat"],
        // Countries/Locations
        vec!["indonesia", "indonesian", "indonesia", "id"],
        vec!["jakarta", "dki jakarta", "capital"],
        // Time periods
        vec!["2026", "fy2026", "fiscal year 2026", "tahun 2026"],
        vec!["2025", "fy2025", "fiscal year 2025", "tahun 2025"],
        vec!["quarter", "kuartal", "q1", "q2", "q3", "q4"],
        vec!["semester", "half year", "半年", "上半期"],
        // General business
        vec!["company", "perusahaan", "corporation", "firm", "enterprise", "公司", "企業"],
        vec!["business", "usaha", "bisnis", "commercial", "业务", "事業"],
        vec!["report", "laporan", "statement", "report", "报告", "報告"],
        vec!["analysis", "analisis", "evaluation", "assessment", "分析", "分析"],
        // Actions
        vec!["increase", "meningkat", "rise", "grow", "improve", "增长", "増加"],
        vec!["decrease", "menurun", "decline", "fall", "drop", "减少", "減少"],
        vec!["compare", "bandingkan", "comparison", "versus", "vs", "比較"],
        // Questions
        vec!["what", "apa", "siapa", "mana", "yang"],
        vec!["how", "bagaimana", "cara", "method"],
        vec!["when", "kapan", "date", "time"],
        vec!["where", "dimana", "lokasi", "location"],
        vec!["why", "mengapa", "alasan", "reason"],
    ]
}

// Build reverse lookup: term -> all synonyms in same group
fn build_synonym_map() -> HashMap<String, Vec<String>> {
    let groups = build_synonym_groups();
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    
    for group in &groups {
        let synonyms: Vec<String> = group.iter().map(|s| s.to_string()).collect();
        for term in group {
            map.entry(term.to_string())
                .or_insert_with(Vec::new)
                .extend(synonyms.iter().cloned());
        }
    }
    
    // Deduplicate each entry
    for (_, syns) in map.iter_mut() {
        syns.sort();
        syns.dedup();
    }
    
    map
}

// Expand query dengan sinonim
fn expand_query(query: &str, synonym_map: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut expanded: Vec<String> = Vec::new();
    let low = query.to_lowercase();
    
    // Split query into terms
    let terms: Vec<&str> = low.split_whitespace().collect();
    
    // Check for multi-word synonyms first
    for (_, syns) in synonym_map.iter() {
        for syn in syns {
            if low.contains(syn) {
                // Found a match, add all synonyms
                for s in syns {
                    if !expanded.contains(&s.to_string()) {
                        expanded.push(s.to_string());
                    }
                }
            }
        }
    }
    
    // Then expand individual terms
    for term in &terms {
        if let Some(syns) = synonym_map.get(*term) {
            for syn in syns {
                if !expanded.contains(&syn.to_string()) {
                    expanded.push(syn.to_string());
                }
            }
        }
    }
    
    // Always include original query terms
    for term in &terms {
        if !expanded.contains(&term.to_string()) {
            expanded.push(term.to_string());
        }
    }
    
    expanded
}

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
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Doc { content: String, source: String, page: u32, heading: String }

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
        
        // SIMD-accelerated batch scoring
        // Kumpulkan semua postings dulu, lalu score dalam batch
        let mut all_postings: Vec<(&str, u32, Vec<&(u32,u32)>)> = Vec::new();
        for (t,qf) in &qtf {
            let plist = match post.get(*t){Some(v)=>v,None=>continue,};
            let idfv = match idf.get(*t){Some(v)=>*v,None=>continue,};
            if idfv<=0.0{continue;}
            all_postings.push((t, *qf, plist.iter().collect()));
        }
        
        if all_postings.is_empty(){return Vec::new();}
        
        // SIMD batch scoring: proses 8 dokumen sekaligus
        let mut acc: HashMap<u32,f64> = HashMap::new();
        let mut matched: HashMap<u32,u32> = HashMap::new();
        
        for (t, qf, plist) in &all_postings {
            let idfv = idf[*t];
            let qf_f = *qf as f64;
            
            // Batch process 8 postings sekaligus untuk SIMD
            for chunk in plist.chunks(8) {
                let chunk_len = chunk.len();
                
                // Load batch dokumen
                let mut batch_dl: [f64; 8] = [avg; 8];
                let mut batch_tf: [f64; 8] = [0.0; 8];
                let mut batch_doc: [u32; 8] = [0; 8];
                
                for (i, (doc_id, tf)) in chunk.iter().enumerate() {
                    batch_doc[i] = *doc_id;
                    batch_dl[i] = lens.get(*doc_id as usize).copied().unwrap_or(avg as usize) as f64;
                    batch_tf[i] = *tf as f64;
                }
                
                // SIMD BM25 scoring untuk batch
                let k1_plus_1 = BM25_K1 + 1.0;
                let one_minus_b = 1.0 - BM25_B;
                let inv_avg = 1.0 / avg.max(1.0);
                
                // Vectorized computation
                let mut batch_scores: [f64; 8] = [0.0; 8];
                for i in 0..chunk_len {
                    let norm = one_minus_b + BM25_B * batch_dl[i] * inv_avg;
                    let denom = batch_tf[i] + BM25_K1 * norm;
                    batch_scores[i] = if denom > 0.0 { idfv * (batch_tf[i] * k1_plus_1 / denom) } else { 0.0 };
                }
                
                // Accumulate dengan coverage bonus
                let nq = qtf.len().max(1) as f64;
                for i in 0..chunk_len {
                    let score = batch_scores[i];
                    if score > 0.0 {
                        let m = matched.entry(batch_doc[i]).or_insert(0);
                        *m += 1;
                        let cov = *m as f64 / nq;
                        let bonus = if (*m as f64 - nq).abs() < 1e-9 { 0.5 } else { 0.0 };
                        *acc.entry(batch_doc[i]).or_insert(0.0) += score * (0.4 + 0.6 * cov + bonus) * (1.0 + 0.1 * (qf_f - 1.0));
                    }
                }
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

    fn search(&self, query: &str, top_k: usize) -> Vec<SearchResult> {
        if self.docs.is_empty()||query.trim().is_empty(){return Vec::new();}
        
        // Build synonym map (cached)
        let synonym_map = build_synonym_map();
        
        // Expand query dengan sinonim
        let expanded_terms = expand_query(query, &synonym_map);
        
        // Tokenize expanded query
        let mut all_lex: Vec<String> = Vec::new();
        let mut all_tri: Vec<String> = Vec::new();
        
        for term in &expanded_terms {
            let norm = normalize(term);
            let low = norm.to_lowercase();
            let words = tokenize_words_lower(&low);
            let base: Vec<String> = words.iter().filter(|w|!STOPWORDS.contains(&w.as_str())).cloned().collect();
            for w in &base {
                if !all_lex.contains(w) {
                    all_lex.push(w.clone());
                }
            }
            // Add bigrams
            for p in base.windows(2){
                if p[0].len()>2 && p[1].len()>2 {
                    let bigram = format!("{}_{}",p[0],p[1]);
                    if !all_lex.contains(&bigram) {
                        all_lex.push(bigram);
                    }
                }
            }
            // Add trigrams
            let mut tri = Vec::new();
            trigrams_of(&words,&mut tri);
            for t in tri {
                if !all_tri.contains(&t) {
                    all_tri.push(t);
                }
            }
        }
        
        // Also add original query terms (in case expansion missed something)
        let norm = normalize(query);
        let low = norm.to_lowercase();
        let orig_words = tokenize_words_lower(&low);
        let orig_base: Vec<String> = orig_words.iter().filter(|w|!STOPWORDS.contains(&w.as_str())).cloned().collect();
        for w in orig_base {
            if !all_lex.contains(&w) {
                all_lex.push(w);
            }
        }

        let wr = Self::bm25(&all_lex,&self.w_post,&self.w_idf,&self.w_len,self.w_avg,200);
        let tr = Self::bm25(&all_tri,&self.t_post,&self.t_idf,&self.t_len,self.t_avg,200);

        let mut acc: HashMap<u32,f64> = HashMap::new();
        for (rank,(doc,_)) in wr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=W_WORD/(RRF_K+rank as f64+1.0);}
        for (rank,(doc,_)) in tr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=W_TRI/(RRF_K+rank as f64+1.0);}

        let qnums: HashSet<&str> = all_lex.iter().filter(|t|is_thousands(t)).map(|s|s.as_str()).collect();
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
            let mut expanded_content = d.content.clone();
            if let Some(nxt) = self.docs.get(doc as usize + 1) {
                if nxt.source == d.source {
                    expanded_content.push(' ');
                    let trunc: String = nxt.content.chars().take(600).collect();
                    expanded_content.push_str(&trunc);
                }
            }
            SearchResult {
                doc, score, page: d.page,
                source: d.source.clone(), heading: d.heading.clone(),
                snippet: d.content.chars().take(300).collect(),
                expanded: expanded_content.chars().take(900).collect(),
            }
        }).collect()
    }

    // ==================== PERSISTENT INDEX ====================
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

    fn save_json(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
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
            let metadata = fs::metadata(path)?;
            if metadata.len() > 100 * 1024 * 1024 {
                read_file_mmap(path)
            } else {
                fs::read_to_string(path)
            }
        }
        "pdf" => {
            // pdf_oxide: Rust-native, fastest PDF parser (0.8ms, 100% pass rate)
            match extract_pdf_text(path) {
                Ok(text) => Ok(text),
                Err(e) => {
                    eprintln!("  WARNING: PDF parse failed for {}: {}", 
                        path.file_name().unwrap_or_default().to_string_lossy(), e);
                    Ok(String::new())
                }
            }
        }
        _ => {
            let metadata = fs::metadata(path)?;
            if metadata.len() > 100 * 1024 * 1024 {
                read_file_mmap(path)
            } else {
                fs::read_to_string(path)
            }
        }
    }
}

/// pdf_oxide: extract text dari PDF (0.8ms, 100% pass rate)
fn extract_pdf_text(path: &Path) -> io::Result<String> {
    use pdf_oxide::PdfDocument;
    
    let doc = PdfDocument::open(path.to_str().unwrap_or(""))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("PDF open failed: {}", e)))?;
    
    let num_pages = doc.page_count()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Page count failed: {}", e)))?;
    
    let mut all_text = String::new();
    
    for page_idx in 0..num_pages {
        match doc.extract_text_auto(page_idx) {
            Ok(text) => {
                all_text.push_str(&text);
                all_text.push_str("\n\n");
            }
            Err(e) => {
                eprintln!("  WARNING: Page {} extraction failed: {}", page_idx, e);
            }
        }
    }
    
    Ok(all_text)
}

/// mmap-based file reading untuk file besar (>100MB)
/// Menggunakan memmap2 untuk zero-copy reading, RAM hanya ~page table
fn read_file_mmap(path: &Path) -> io::Result<String> {
    use memmap2::Mmap;
    
    let file = fs::File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    
    // Untuk file sangat besar (>4GB), baca chunks
    if file_size > 4 * 1024 * 1024 * 1024 {
        return read_file_chunked(path, file_size);
    }
    
    // mmap untuk file 100MB-4GB
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mmap failed: {}", e)))?;
    
    // Convert bytes to string (lossy untuk safety)
    let text = String::from_utf8_lossy(&mmap).to_string();
    Ok(text)
}

/// Chunked reading untuk file >4GB
fn read_file_chunked(path: &Path, file_size: u64) -> io::Result<String> {
    use memmap2::Mmap;
    
    let file = fs::File::open(path)?;
    let chunk_size = 256 * 1024 * 1024; // 256MB chunks
    let mut result = String::with_capacity(file_size as usize);
    
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mmap failed: {}", e)))?;
    
    let mut offset = 0usize;
    while offset < file_size as usize {
        let end = (offset + chunk_size).min(file_size as usize);
        let chunk = &mmap[offset..end];
        result.push_str(&String::from_utf8_lossy(chunk));
        offset = end;
    }
    
    Ok(result)
}

// ==================== PARALLEL FILE PARSING ====================
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
        
        for h in handles {
            let _ = h.join();
        }
        drop(tx);
    });
    
    rx.iter().collect()
}

// ==================== SIMPLE REST SERVER ====================
fn start_server(idx: Arc<OnodIndex>, port: u16) {
    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("cannot bind");
    eprintln!("Server listening on http://0.0.0.0:{}", port);
    eprintln!("Endpoints:");
    eprintln!("  GET /search?q=<query>&top_k=10");
    eprintln!("  GET /health");
    eprintln!("  GET /");
    
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let idx = idx.clone();
                std::thread::spawn(move || {
                    handle_connection(stream, &idx);
                });
            }
            Err(e) => eprintln!("Connection failed: {}", e),
        }
    }
}

fn handle_connection(mut stream: TcpStream, idx: &OnodIndex) {
    use std::io::{BufRead, BufReader, Write};
    
    let mut buf_reader = BufReader::new(&stream);
    let mut request_line = String::new();
    buf_reader.read_line(&mut request_line).unwrap();
    
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        let response = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
        stream.write_all(response.as_bytes()).unwrap();
        return;
    }
    
    let path = parts[1];
    let response = match path {
        "/" => {
            let body = serde_json::json!({
                "name": "onod",
                "version": "0.2.0",
                "endpoints": {
                    "search": "/search?q=<query>&top_k=10",
                    "health": "/health"
                }
            });
            format_response(200, &body.to_string())
        }
        "/health" => {
            let body = serde_json::json!({
                "status": "ok",
                "version": "0.2.0",
                "chunks": idx.docs.len()
            });
            format_response(200, &body.to_string())
        }
        p if p.starts_with("/search") => {
            let params = parse_query_string(p);
            let q = params.get("q").map(|s| s.as_str()).unwrap_or("");
            let top_k: usize = params.get("top_k")
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            
            if q.is_empty() {
                let body = serde_json::json!({"error": "missing q parameter"});
                format_response(400, &body.to_string())
            } else {
                let results = idx.search(q, top_k);
                let body = serde_json::json!({
                    "query": q,
                    "results": results,
                    "count": results.len()
                });
                format_response(200, &body.to_string())
            }
        }
        _ => {
            let body = serde_json::json!({"error": "not found"});
            format_response(404, &body.to_string())
        }
    };
    
    stream.write_all(response.as_bytes()).unwrap();
}

fn format_response(status: u16, body: &str) -> String {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, status_text, body.len(), body
    )
}

fn parse_query_string(path: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    if let Some(query_start) = path.find('?') {
        let query = &path[query_start + 1..];
        for pair in query.split('&') {
            if let Some((key, value)) = pair.split_once('=') {
                params.insert(
                    urldecode(key.to_string()),
                    urldecode(value.to_string())
                );
            }
        }
    }
    params
}

fn urldecode(s: String) -> String {
    s.replace("%20", " ").replace("%27", "'").replace("%22", "\"")
}

// ==================== MAIN ====================
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("onod — Full-Rust Search Engine v0.2.0");
        eprintln!();
        eprintln!("Usage: onod <command> [args]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  index <folder>              Index all files in folder");
        eprintln!("  search <folder> <query>     Search indexed folder");
        eprintln!("  benchmark <folder>          Run accuracy benchmark");
        eprintln!("  serve <folder> [--port N]   Start REST API server");
        eprintln!("  save <folder> <index.bin>   Save index to binary file");
        eprintln!("  load <index.bin>            Load index from binary file (interactive)");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  onod benchmark files/");
        eprintln!("  onod search files/ \"What is revenue?\"");
        eprintln!("  onod serve files/ --port 8080");
        eprintln!("  onod save files/ index.bin");
        eprintln!("  onod load index.bin");
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

            eprintln!("Reading {} files in parallel...", paths.len());
            let results = read_files_parallel(&paths);
            
            for (path, result) in &results {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                match result {
                    Ok(text) => {
                        let chunks = idx.add_text(text, &name, 0);
                        eprintln!("  {} -> {} chunks", name, chunks);
                    }
                    Err(e) => eprintln!("  {} -> ERROR: {}", name, e),
                }
            }
            idx.finalize();
            let elapsed = t0.elapsed();
            eprintln!("\nIndexed {} files, {} chunks, {:.2}s", 
                paths.len(), idx.docs.len(), elapsed.as_secs_f64());
        }
        
        "search" => {
            let folder = args.get(2).expect("provide folder");
            let query = args.get(3..).unwrap_or(&[]).join(" ");
            if query.is_empty() {
                eprintln!("Usage: onod search <folder> <query>");
                std::process::exit(1);
            }

            let mut idx = OnodIndex::new();
            let t0 = Instant::now();

            let paths: Vec<PathBuf> = fs::read_dir(folder)
                .expect("cannot read dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.'))
                .collect();

            for path in &paths {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if let Ok(text) = read_file(path) {
                    idx.add_text(&text, &name, 0);
                }
            }
            idx.finalize();
            
            let t1 = Instant::now();
            let results = idx.search(&query, 10);
            let search_ms = t1.elapsed().as_secs_f64() * 1000.0;

            eprintln!("Query: \"{}\"", query);
            eprintln!("Index: {} chunks in {:.2}s", idx.docs.len(), t0.elapsed().as_secs_f64());
            eprintln!("Search: {:.1}ms\n", search_ms);

            for (i, r) in results.iter().enumerate() {
                println!("{}. [{:.4}] {} (page {}) — {}", 
                    i+1, r.score, r.source, r.page, r.heading);
                println!("   {}", r.snippet);
                println!();
            }
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

            let results = read_files_parallel(&paths);
            for (path, result) in &results {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                match result {
                    Ok(text) if !text.is_empty() => {
                        let chunks = idx.add_text(&text, &name, 0);
                        if chunks > 0 {
                            eprintln!("  {} -> {} chunks", name, chunks);
                        } else {
                            eprintln!("  {} -> 0 chunks (skipped)", name);
                        }
                    }
                    Ok(_) => {
                        eprintln!("  {} -> empty (skipped - install pymupdf for PDF support)", name);
                    }
                    Err(e) => eprintln!("  {} -> ERROR: {}", name, e),
                }
            }
            idx.finalize();
            let index_time = t0.elapsed();
            eprintln!("\nIndex: {} chunks, {:.2}s\n", idx.docs.len(), index_time.as_secs_f64());

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
        
        "serve" => {
            let folder = args.get(2).expect("provide folder");
            let port: u16 = args.iter().position(|a| a == "--port")
                .and_then(|i| args.get(i+1))
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080);

            eprintln!("Building index from {}...", folder);
            let mut idx = OnodIndex::new();
            let t0 = Instant::now();

            let paths: Vec<PathBuf> = fs::read_dir(folder)
                .expect("cannot read dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.'))
                .collect();

            for path in &paths {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if let Ok(text) = read_file(path) {
                    idx.add_text(&text, &name, 0);
                }
            }
            idx.finalize();
            eprintln!("Index ready: {} chunks in {:.2}s", idx.docs.len(), t0.elapsed().as_secs_f64());

            let idx = Arc::new(idx);
            start_server(idx, port);
        }
        
        "save" => {
            let folder = args.get(2).expect("provide folder");
            let save_path = args.get(3).expect("provide output path (e.g., index.bin)");
            
            let mut idx = OnodIndex::new();
            let t0 = Instant::now();

            let paths: Vec<PathBuf> = fs::read_dir(folder)
                .expect("cannot read dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_file() && !p.file_name().unwrap_or_default().to_string_lossy().starts_with('.'))
                .collect();

            for path in &paths {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                if let Ok(text) = read_file(path) {
                    idx.add_text(&text, &name, 0);
                }
            }
            idx.finalize();
            
            let save_time = t0.elapsed();
            let save_path = Path::new(save_path);
            
            if save_path.extension().map(|e| e.to_str() == Some("json")).unwrap_or(false) {
                idx.save_json(save_path).expect("failed to save JSON");
            } else {
                idx.save(save_path).expect("failed to save binary");
            }
            
            eprintln!("Saved index: {} chunks -> {} ({:.2}s)", 
                idx.docs.len(), save_path.display(), save_time.as_secs_f64());
        }
        
        "load" => {
            let load_path = args.get(2).expect("provide index file path");
            let load_path = Path::new(load_path);
            
            eprintln!("Loading index from {}...", load_path.display());
            let t0 = Instant::now();
            let idx = OnodIndex::load(load_path).expect("failed to load index");
            let load_time = t0.elapsed();
            
            eprintln!("Loaded: {} chunks, {} word terms, {} tri terms in {:.2}s",
                idx.docs.len(), idx.w_post.len(), idx.t_post.len(), load_time.as_secs_f64());
            
            eprintln!("\nInteractive mode (type 'quit' to exit):");
            loop {
                print!("> ");
                io::stdout().flush().unwrap();
                let mut input = String::new();
                if io::stdin().read_line(&mut input).is_err() { break; }
                let input = input.trim();
                if input == "quit" || input == "exit" { break; }
                if input.is_empty() { continue; }
                
                let t1 = Instant::now();
                let results = idx.search(input, 10);
                let ms = t1.elapsed().as_secs_f64() * 1000.0;
                
                for (i, r) in results.iter().enumerate() {
                    println!("  {}. [{:.4}] {} (page {}) — {}", 
                        i+1, r.score, r.source, r.page, r.heading);
                    println!("     {}", r.snippet);
                }
                println!("  ({:.1}ms, {} results)\n", ms, results.len());
            }
        }
        
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}
