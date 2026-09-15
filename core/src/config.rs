use std::fs;

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Config {
    /// Jumlah karakter per chunk saat indexing. Semakin besar → fewer chunks → faster indexing,
    /// tapi konteks per chunk lebih luas (bisa kurang presisi untuk pertanyaan spesifik).
    /// sweet spot: 2400-4800 untuk financial docs.
    pub chunk_chars: usize,

    /// Karakter overlap antar chunk. Memastikan informasi di boundary chunk tidak hilang.
    /// Biasanya 10-20% dari chunk_chars.
    pub chunk_overlap: usize,

    /// Minimum karakter agar chunk dianggap valid. Chunk lebih kecil dari ini dibuang.
    pub min_chunk: usize,

    /// Hard split: paksa split kalau chunk melebihi ini (misalnya satu paragraf sangat panjang).
    /// Biasanya 1.5x chunk_chars.
    pub hard_split: usize,

    /// Dimensi embedding output dari model. MiniLM = 384, larger models bisa 768.
    pub embedding_dim: usize,

    /// Jumlah kandidat dari retrieval awal (dense + sparse) sebelum reranking.
    /// Semakin besar → lebih lengkap tapi lebih lambat. 200 cukup untuk korpus <10k chunks.
    pub top_k_candidates: usize,

    /// Jumlah hasil akhir yang dikembalikan ke user (default di search/serve).
    pub top_k_results: usize,

    /// Bobot reranking (0.0-1.0). Semakin tinggi → semakin banyak pengaruh reranker.
    /// 0.6 = 60% reranker, 40% ranking awal.
    pub rerank_weight: f64,

    /// Financial boost: skor tambahan untuk dokumen yang mengandung istilah keuangan
    /// (revenue, laba, aset, dll). Membantu query financial lebih relevan.
    /// 0.0 = tidak ada boost, 1.0 = boost maksimal.
    pub financial_boost: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_chars: 4800,
            chunk_overlap: 400,
            min_chunk: 100,
            hard_split: 7200,
            embedding_dim: 384,
            top_k_candidates: 200,
            top_k_results: 10,
            rerank_weight: 0.6,
            financial_boost: 0.3,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        for path in &["onod.json", "config.json"] {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(c) = serde_json::from_str::<Config>(&content) {
                    eprintln!("Loaded config from {}", path);
                    return c;
                }
            }
        }
        Config::default()
    }
}

static CONFIG: std::sync::OnceLock<Config> = std::sync::OnceLock::new();

pub fn get_config() -> &'static Config {
    CONFIG.get_or_init(|| Config::load())
}
