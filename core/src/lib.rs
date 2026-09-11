use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use unicode_normalization::UnicodeNormalization;

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

fn is_thousands(tok: &str) -> bool {
    let b = tok.as_bytes();
    let mut i = 0;
    let mut head = 0;
    while i < b.len() && b[i].is_ascii_digit() && head < 3 { i += 1; head += 1; }
    if head == 0 || head > 3 { return false; }
    let mut groups = 0;
    while i + 4 <= b.len() && b[i] == b'.' && b[i+1].is_ascii_digit() && b[i+2].is_ascii_digit() && b[i+3].is_ascii_digit() {
        i += 4; groups += 1;
    }
    if groups == 0 { return false; }
    if i < b.len() && b[i] == b',' {
        i += 1; let s = i;
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
        if i == s { return false; }
    }
    i == b.len()
}

struct Analyzed { lex: Vec<String>, tri: Vec<String> }

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

fn analyze_text(norm: &str) -> Analyzed {
    let low = norm.to_lowercase();
    let words = tokenize_words_lower(&low);
    let base: Vec<String> = words.iter().filter(|w| !STOPWORDS.contains(&w.as_str())).cloned().collect();
    let mut lex = base.clone();
    for p in base.windows(2) { if p[0].len()>2 && p[1].len()>2 { lex.push(format!("{}_{}",p[0],p[1])); } }
    let mut tri = Vec::new();
    trigrams_of(&words, &mut tri);
    Analyzed { lex, tri }
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
    if start < ch.len() {
        let s: String = ch[start..].iter().collect();
        if !s.trim().is_empty() { parts.push(s.trim().to_string()); }
    }
    let mut out = Vec::with_capacity(parts.len());
    for s in parts {
        let mut rest = s.as_str();
        while rest.len() > HARD_SPLIT {
            let mut cut = HARD_SPLIT.min(rest.len());
            if let Some(sp) = rest[..cut].rfind(' ') { if sp>300{cut=sp;} }
            while cut>0 && !rest.is_char_boundary(cut){cut-=1;}
            out.push(rest[..cut].trim().to_string());
            rest = rest[cut..].trim();
        }
        if !rest.is_empty() { out.push(rest.to_string()); }
    }
    out
}

struct Chunk { content: String, heading: String }

fn chunk_text(text: &str) -> Vec<Chunk> {
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
            if content.len() >= MIN_CHUNK { chunks.push(Chunk{content,heading:heading.clone()}); }
            let mut tail = Vec::new(); let mut tl = 0;
            for item in buf.iter().rev() { tail.insert(0,item.clone()); tl+=item.len()+1; if tl>=CHUNK_OVERLAP||tail.len()>=2{break;} }
            buf = tail; buf_len = tl;
        }
        buf_len += s.len() + 1;
        buf.push(s);
    }
    if !buf.is_empty() {
        let content = buf.join(" ");
        if content.len() >= MIN_CHUNK { chunks.push(Chunk{content,heading}); }
    }
    chunks
}

struct Doc { content: String, source: String, page: u32, heading: String }

pub struct OnodIndex {
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
        let analyzed: Vec<(Chunk,Analyzed)> = chunks.into_par_iter().map(|c|{let nn=normalize(&c.content); let a=analyze_text(&nn);(c,a)}).collect();
        let mut n = 0u32;
        for (c,a) in analyzed {
            if a.lex.is_empty() && a.tri.is_empty(){continue;}
            let id = self.docs.len() as u32;
            self.docs.push(Doc{content:c.content,source:source.to_string(),page,heading:c.heading});
            self.w_len.push(a.lex.len()); self.t_len.push(a.tri.len());
            let mut tf:HashMap<&str,u32> = HashMap::new();
            for t in &a.lex{*tf.entry(t.as_str()).or_insert(0)+=1;}
            for (t,f) in tf{self.w_post.entry(t.to_string()).or_default().push((id,f));}
            let mut tf2:HashMap<&str,u32> = HashMap::new();
            for t in &a.tri{*tf2.entry(t.as_str()).or_insert(0)+=1;}
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
        let mut scored: Vec<(u32,f64)> = acc.into_par_iter().map(|(doc,s)|{
            let m=matched.get(&doc).copied().unwrap_or(0) as f64;
            let cov=m/nq;
            let bonus=if (m-nq).abs()<1e-9{0.5}else{0.0};
            (doc,s*(0.4+0.6*cov+bonus))
        }).collect();
        let k = top_k.min(scored.len());
        scored.select_nth_unstable_by(k.saturating_sub(1),|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    fn search(&self, query: &str, top_k: usize) -> Vec<(u32,f64)> {
        if self.docs.is_empty()||query.trim().is_empty(){return Vec::new();}
        let norm = normalize(query);
        let low = norm.to_lowercase();
        let words = tokenize_words_lower(&low);
        let base: Vec<String> = words.iter().filter(|w|!STOPWORDS.contains(&w.as_str())).cloned().collect();
        let mut lex = base.clone();
        for p in base.windows(2){if p[0].len()>2&&p[1].len()>2{lex.push(format!("{}_{}",p[0],p[1]));}}
        let mut tri = Vec::new();
        trigrams_of(&words,&mut tri);
        let (wr,tr) = rayon::join(
            ||Self::bm25(&lex,&self.w_post,&self.w_idf,&self.w_len,self.w_avg,200),
            ||Self::bm25(&tri,&self.t_post,&self.t_idf,&self.t_len,self.t_avg,200),
        );
        let mut acc: HashMap<u32,f64> = HashMap::new();
        for (rank,(doc,_)) in wr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=W_WORD/(RRF_K+rank as f64+1.0);}
        for (rank,(doc,_)) in tr.iter().enumerate(){*acc.entry(*doc).or_insert(0.0)+=W_TRI/(RRF_K+rank as f64+1.0);}
        let qnums: HashSet<&str> = lex.iter().filter(|t|is_thousands(t)).map(|s|s.as_str()).collect();
        let mut fused: Vec<(u32,f64)> = acc.into_iter().collect();
        if !qnums.is_empty(){for(doc,s) in fused.iter_mut(){let c=&self.docs[*doc as usize].content; for qn in &qnums{if c.contains(*qn){*s+=0.6;break;}}}}
        fused.sort_by(|a,b|b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(top_k);
        fused
    }
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len()+2);
    for c in s.chars() {
        if c == '"' { o.push('\\'); o.push('"'); }
        else if c == '\\' { o.push('\\'); o.push('\\'); }
        else if c == '\n' { o.push('\\'); o.push('n'); }
        else if c == '\r' { o.push('\\'); o.push('r'); }
        else if c == '\t' { o.push('\\'); o.push('t'); }
        else if (c as u32) < 0x20 { o.push_str(&format!("\\u{:04x}", c as u32)); }
        else { o.push(c); }
    }
    o
}

fn snippet(s: &str, n: usize) -> String {
    let mut o = String::new();
    for (i,c) in s.chars().enumerate(){if i>=n{break;} o.push(if c.is_whitespace(){' '}else{c});}
    o
}

const MAGIC: &[u8;8] = b"ONODIDX1";
fn w_u32(v: u32, o: &mut Vec<u8>) { o.extend_from_slice(&v.to_le_bytes()); }
fn w_str(s: &str, o: &mut Vec<u8>) { w_u32(s.len() as u32, o); o.extend_from_slice(s.as_bytes()); }

struct Reader<'a>{b:&'a[u8],p:usize}
impl<'a> Reader<'a> {
    fn u32(&mut self)->Option<u32>{if self.p+4>self.b.len(){return None;} let v=u32::from_le_bytes(self.b[self.p..self.p+4].try_into().ok()?); self.p+=4; Some(v)}
    fn read_str(&mut self)->Option<String>{let n=self.u32()? as usize; if self.p+n>self.b.len(){return None;} let s=std::str::from_utf8(&self.b[self.p..self.p+n]).ok()?.to_string(); self.p+=n; Some(s)}
}

impl OnodIndex {
    fn save_to(&self, path: &str) -> bool {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        w_u32(self.docs.len() as u32,&mut o);
        for d in &self.docs{w_str(&d.content,&mut o);w_str(&d.source,&mut o);w_u32(d.page,&mut o);w_str(&d.heading,&mut o);}
        w_u32(self.w_len.len() as u32,&mut o);for v in &self.w_len{w_u32(*v as u32,&mut o);}
        w_u32(self.t_len.len() as u32,&mut o);for v in &self.t_len{w_u32(*v as u32,&mut o);}
        for post in [&self.w_post,&self.t_post]{
            w_u32(post.len() as u32,&mut o);
            let mut terms:Vec<_>=post.iter().collect();terms.sort_by(|a,b|a.0.cmp(b.0));
            for(t,plist) in terms{w_str(t,&mut o);w_u32(plist.len() as u32,&mut o);for(doc,tf) in plist{w_u32(*doc,&mut o);w_u32(*tf,&mut o);}}
        }
        std::fs::write(path,&o).is_ok()
    }

    fn load_from(path: &str) -> Option<Self> {
        let b = std::fs::read(path).ok()?;
        let mut r = Reader{b:&b,p:0};
        if r.b.len()<8||&r.b[0..8]!=MAGIC{return None;}
        r.p=8;
        let nd=r.u32()? as usize;
        let mut idx=OnodIndex::new();
        for _ in 0..nd{let content=r.read_str()?;let source=r.read_str()?;let page=r.u32()?;let heading=r.read_str()?;idx.docs.push(Doc{content,source,page,heading});}
        let nw=r.u32()? as usize; for _ in 0..nw{idx.w_len.push(r.u32()? as usize);}
        let nt=r.u32()? as usize; for _ in 0..nt{idx.t_len.push(r.u32()? as usize);}
        for post in [&mut idx.w_post,&mut idx.t_post]{
            let n=r.u32()? as usize;
            for _ in 0..n{let t=r.read_str()?;let m=r.u32()? as usize;let mut v=Vec::with_capacity(m);for _ in 0..m{v.push((r.u32()?,r.u32()?));}post.insert(t,v);}
        }
        idx.finalize();
        Some(idx)
    }
}

fn cstr(ptr: *const c_char) -> String {
    if ptr.is_null(){return String::new();}
    unsafe{CStr::from_ptr(ptr).to_string_lossy().into_owned()}
}

#[no_mangle]
pub extern "C" fn onod_new() -> *mut OnodIndex { Box::into_raw(Box::new(OnodIndex::new())) }

#[no_mangle]
pub extern "C" fn onod_free(idx: *mut OnodIndex) { if !idx.is_null(){unsafe{drop(Box::from_raw(idx));}} }

#[no_mangle]
pub extern "C" fn onod_free_str(s: *mut c_char) { if !s.is_null(){unsafe{drop(CString::from_raw(s));}} }

#[no_mangle]
pub extern "C" fn onod_add(idx: *mut OnodIndex, text_ptr: *const c_char, src_ptr: *const c_char, page: u32) -> u32 {
    if idx.is_null(){return 0;}
    let text=cstr(text_ptr); let src=cstr(src_ptr);
    unsafe{(*idx).add_text(&text,&src,page)}
}

#[no_mangle]
pub extern "C" fn onod_finalize(idx: *mut OnodIndex) { if !idx.is_null(){unsafe{(*idx).finalize();}} }

#[no_mangle]
pub extern "C" fn onod_search(idx: *const OnodIndex, q_ptr: *const c_char, top_k: u32, out_len: *mut usize) -> *mut c_char {
    if idx.is_null(){return CString::new("[]").unwrap().into_raw();}
    let q=cstr(q_ptr);
    let res=unsafe{(*idx).search(&q,top_k.max(1) as usize)};
    let idxr=unsafe{&*idx};
    let mut o=String::from("[");
    for(i,(doc,score)) in res.iter().enumerate(){
        let d=&idxr.docs[*doc as usize];
        let mut expanded=d.content.clone();
        if let Some(nxt)=idxr.docs.get(*doc as usize+1){if nxt.source==d.source{expanded.push(' ');expanded.push_str(&snippet(&nxt.content,600));}}
        if i>0{o.push(',');}
        o.push_str("{\"doc\":");
        o.push_str(&doc.to_string());
        o.push_str(",\"score\":");
        o.push_str(&format!("{:.4}",score));
        o.push_str(",\"page\":");
        o.push_str(&d.page.to_string());
        o.push_str(",\"source\":\"");
        o.push_str(&json_escape(&d.source));
        o.push_str("\",\"heading\":\"");
        o.push_str(&json_escape(&d.heading));
        o.push_str("\",\"snippet\":\"");
        o.push_str(&json_escape(&snippet(&d.content,300)));
        o.push_str("\",\"expanded\":\"");
        o.push_str(&json_escape(&snippet(&expanded,900)));
        o.push_str("\"}");
    }
    o.push(']');
    let c=CString::new(o).unwrap_or_default();
    if !out_len.is_null(){unsafe{*out_len=c.as_bytes().len();}}
    c.into_raw()
}

#[no_mangle]
pub extern "C" fn onod_stats(idx: *const OnodIndex, out_len: *mut usize) -> *mut c_char {
    if idx.is_null(){return CString::new("{}").unwrap().into_raw();}
    let r=unsafe{&*idx};
    let s=format!("{{\"chunks\":{},\"word_terms\":{},\"tri_terms\":{}}}",r.docs.len(),r.w_post.len(),r.t_post.len());
    let c=CString::new(s).unwrap_or_default();
    if !out_len.is_null(){unsafe{*out_len=c.as_bytes().len();}}
    c.into_raw()
}

#[no_mangle]
pub extern "C" fn onod_save(idx: *const OnodIndex, path_ptr: *const c_char) -> bool {
    if idx.is_null()||path_ptr.is_null(){return false;}
    let path=cstr(path_ptr);
    unsafe{(*idx).save_to(&path)}
}

#[no_mangle]
pub extern "C" fn onod_load(path_ptr: *const c_char) -> *mut OnodIndex {
    if path_ptr.is_null(){return std::ptr::null_mut();}
    let path=cstr(path_ptr);
    match OnodIndex::load_from(&path){Some(idx)=>Box::into_raw(Box::new(idx)),None=>std::ptr::null_mut(),}
}
