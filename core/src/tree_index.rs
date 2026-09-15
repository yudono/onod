use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

use crate::config::get_config;
use crate::TransformerEmbedder;
use crate::SparseEmbedder;

/// TreeNode: hierarchical document with paragraph/chapter structure.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct TreeNode {
    pub id: u32,
    pub level: u8,
    pub heading: String,
    pub content: String,
    pub children: Vec<u32>,
    pub source: String,
    pub page: u32,
    #[serde(skip)]
    pub dense_embedding: Vec<f32>,
    #[serde(skip)]
    pub sparse_embedding: HashMap<u32, f32>,
}

pub struct TreeIndex {
    pub nodes: Vec<TreeNode>,
    pub root_children: Vec<u32>,
    pub dim_df: Vec<u32>,
}

impl TreeIndex {
    pub fn new() -> Self {
        let dim = get_config().embedding_dim;
        Self { nodes: Vec::new(), root_children: Vec::new(), dim_df: vec![0u32; dim] }
    }

    pub fn build_from_text(&mut self, text: &str, source: &str) {
        let cfg = get_config();
        let norm: String = text.nfc().collect();
        let lines: Vec<&str> = norm.lines().collect();
        let mut stack: Vec<(u8, u32)> = Vec::new();
        let mut para_buf: Vec<String> = Vec::new();
        let mut current_heading = String::new();

        let mut flush_para = |buf: &mut Vec<String>, heading: &str, nodes: &mut Vec<TreeNode>, stack: &[(u8, u32)]| {
            if buf.is_empty() { return; }
            let text = buf.join(" ");
            if text.len() < cfg.min_chunk { buf.clear(); return; }
            let id = nodes.len() as u32;
            nodes.push(TreeNode {
                id, level: 0, heading: heading.to_string(), content: text,
                children: Vec::new(), source: source.to_string(), page: 0,
                dense_embedding: Vec::new(), sparse_embedding: HashMap::new(),
            });
            if let Some(&(_, pid)) = stack.last() { nodes[pid as usize].children.push(id); }
            buf.clear();
        };

        for line in &lines {
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            let level = detect_level(trimmed);
            if level > 0 && level <= 2 {
                flush_para(&mut para_buf, &current_heading, &mut self.nodes, &stack);
                while let Some(&(lvl, _)) = stack.last() { if lvl < level { break; } stack.pop(); }
                let id = self.nodes.len() as u32;
                let heading: String = trimmed.chars().take(200).collect();
                self.nodes.push(TreeNode {
                    id, level, heading: heading.clone(), content: String::new(),
                    children: Vec::new(), source: source.to_string(), page: 0,
                    dense_embedding: Vec::new(), sparse_embedding: HashMap::new(),
                });
                if let Some(&(_, pid)) = stack.last() { self.nodes[pid as usize].children.push(id); }
                else { self.root_children.push(id); }
                stack.push((level, id));
                current_heading = heading;
            } else {
                para_buf.push(trimmed.to_string());
            }
        }
        flush_para(&mut para_buf, &current_heading, &mut self.nodes, &stack);

        if self.nodes.is_empty() {
            let chunks = crate::chunk_text(&norm);
            for (content, heading) in chunks {
                if content.len() < cfg.min_chunk { continue; }
                let id = self.nodes.len() as u32;
                self.root_children.push(id);
                self.nodes.push(TreeNode {
                    id, level: 1, heading, content,
                    children: Vec::new(), source: source.to_string(), page: 0,
                    dense_embedding: Vec::new(), sparse_embedding: HashMap::new(),
                });
            }
        }

        // Fill parent content = own + children
        let ids: Vec<u32> = self.nodes.iter().map(|n| n.id).collect();
        for id in ids.into_iter().rev() {
            let children = self.nodes[id as usize].children.clone();
            let mut full = self.nodes[id as usize].content.clone();
            for cid in &children {
                let ct = self.nodes[*cid as usize].content.clone();
                if !full.is_empty() && !ct.is_empty() { full.push(' '); }
                full.push_str(&ct);
            }
            self.nodes[id as usize].content = full;
        }
    }

    /// Sparse embed semua nodes (instant).
    pub fn build_sparse(&mut self, sparse_embedder: &SparseEmbedder) {
        for node in &mut self.nodes {
            node.sparse_embedding = sparse_embedder.embed(&node.content);
        }
    }

    /// Dense embed SEMUA nodes dalam batch besar. ~200-500 nodes → ~2-5s.
    pub fn embed_all_dense(&mut self, embedder: &TransformerEmbedder, progress: Option<&dyn Fn(usize, usize)>) {
        let total = self.nodes.len();
        if total == 0 { return; }
        let texts: Vec<String> = self.nodes.iter().map(|n| n.content.clone()).collect();

        let has_model = embedder.has_model();
        let dense_embeddings: Vec<Vec<f32>> = if has_model {
            if let Ok(mut guard) = embedder.model.lock() {
                if let Some(ref mut model) = *guard {
                    if let Some(cb) = progress { cb(0, total); }
                    let batch_size = 512;
                    let mut all = Vec::with_capacity(total);
                    for batch_start in (0..total).step_by(batch_size) {
                        let batch_end = (batch_start + batch_size).min(total);
                        if let Some(cb) = progress { cb(batch_start, total); }
                        let batch: Vec<String> = texts[batch_start..batch_end].to_vec();
                        let emb = crate::TransformerEmbedder::embed_batch_static(model, &embedder.tokenizer, &batch, embedder.dim);
                        all.extend(emb);
                    }
                    all
                } else { texts.iter().map(|t| embedder.embed(t)).collect() }
            } else { texts.iter().map(|t| embedder.embed(t)).collect() }
        } else { texts.iter().map(|t| embedder.embed(t)).collect() };

        for (i, dense) in dense_embeddings.into_iter().enumerate() {
            for (j, v) in dense.iter().enumerate() {
                if *v > 0.0 {
                    if j >= self.dim_df.len() { self.dim_df.resize(j + 1, 0); }
                    self.dim_df[j] += 1;
                }
            }
            self.nodes[i].dense_embedding = dense;
        }
        if let Some(cb) = progress { cb(total, total); }
    }

    /// Search: sparse top-50 → dense rerank.
    pub fn search(&self, query: &str, embedder: &TransformerEmbedder, sparse_embedder: &SparseEmbedder, top_k: usize) -> Vec<(u32, f64, String)> {
        if self.nodes.is_empty() || query.trim().is_empty() { return Vec::new(); }
        let cfg = get_config();
        let query_sparse = sparse_embedder.embed(query);

        // Stage 1: Sparse scoring — top 50
        let mut candidates: Vec<(u32, f64)> = self.nodes.iter().enumerate()
            .filter(|(_, n)| !n.sparse_embedding.is_empty())
            .map(|(i, n)| (i as u32, crate::SparseEmbedder::sparse_cosine(&query_sparse, &n.sparse_embedding)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(50);

        // Stage 2: Dense rerank
        let query_dense = embedder.embed_query(query, &self.dim_df, self.nodes.len());
        let mut results: Vec<(u32, f64)> = candidates.into_iter().map(|(id, sparse_score)| {
            let node = &self.nodes[id as usize];
            let dense_score = if !node.dense_embedding.is_empty() {
                crate::cosine_similarity(&query_dense, &node.dense_embedding)
            } else { 0.0 };
            let fin = if node.content.to_lowercase().contains("revenue") || node.content.to_lowercase().contains("pendapatan") { cfg.financial_boost } else { 0.0 };
            (id, 0.4 * dense_score + 0.6 * sparse_score + fin)
        }).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results.iter().take(top_k).map(|&(id, score)| {
            let node = &self.nodes[id as usize];
            (id, score, node.content.chars().take(300).collect::<String>())
        }).collect()
    }
}

fn detect_level(line: &str) -> u8 {
    let lower = line.to_lowercase();
    let trimmed = line.trim();
    if lower.starts_with("bab ") || lower.starts_with("chapter ") || lower.starts_with("bagian ") { return 1; }
    if let Some(fc) = trimmed.chars().next() {
        if fc.is_ascii_digit() {
            let rest: String = trimmed.chars().take(20).collect();
            if let Some(dp) = rest.find('.') {
                let after = rest[dp+1..].trim_start();
                if after.len() >= 5 && !after.starts_with(|c: char| c.is_ascii_digit()) { return 2; }
            }
        }
    }
    0
}
