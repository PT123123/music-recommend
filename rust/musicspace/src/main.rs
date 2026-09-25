//! Portability probe: the online scoring path in std-only Rust.
//!
//! Reads the export written by `scripts/bench_portability.py`, re-derives song->song
//! rankings from the raw per-track payload, and diffs them against Python's own top-20.
//! Every weight, group definition and blend comes from the manifest, so this binary
//! follows `configs/weights.yaml` without code changes -- the point is to find out
//! whether the *whole* query side (percentiles, sequence similarity, weighted score,
//! re-rank) can live outside Python, and what it costs.
//!
//!   cargo run --release -- data/portability
//!   cargo run --release -- data/portability --verify

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::process::exit;
use std::time::Instant;

const SEP: char = '\u{1f}'; // field-internal separator used by the exporter

#[derive(Clone)]
struct Track {
    id: String,
    scalars: Vec<Option<f64>>,
    seqs: Vec<Vec<u32>>,
    curve: Vec<f64>,
    has_vocal: bool,
    emb: Vec<f64>,
}

struct Cfg {
    groups: BTreeMap<String, Vec<usize>>,           // group -> scalar column indices
    seq_of_group: BTreeMap<String, usize>,           // group -> sequence column index
    weights: Vec<(String, f64)>,                     // manifest order == Python dict order
    blend_scalar_sequence: f64,
    curve_blend: (f64, f64),
    seq_weights: [f64; 4],
    ngram_n: usize,
    limit: usize,
}

struct Space {
    tracks: Vec<Track>,
    sorted: Vec<Vec<f64>>, // per scalar column: all present values, sorted
    cfg: Cfg,
}

fn parse_opt(s: &str) -> Option<f64> {
    if s.is_empty() {
        None
    } else {
        s.parse::<f64>().ok()
    }
}

fn kv_list(s: &str) -> Vec<String> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect()
}

fn load(dir: &str) -> Space {
    let manifest = fs::read_to_string(format!("{dir}/manifest.kv")).expect("manifest.kv: run scripts/bench_portability.py first");
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for line in manifest.lines() {
        if let Some((k, v)) = line.split_once('\t') {
            map.insert(k.to_string(), v.to_string());
        }
    }

    let scalar_cols = kv_list(map.get("scalar_cols").expect("scalar_cols"));
    let seq_cols = kv_list(map.get("seq_cols").expect("seq_cols"));
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for entry in map.get("feature_groups").expect("feature_groups").split('|') {
        let (g, cols) = entry.split_once(':').expect("group:cols");
        let idx = kv_list(cols)
            .iter()
            .map(|c| scalar_cols.iter().position(|x| x == c).unwrap_or_else(|| panic!("unknown column {c}")))
            .collect();
        groups.insert(g.to_string(), idx);
    }
    let mut seq_of_group: BTreeMap<String, usize> = BTreeMap::new();
    for entry in map.get("sequence_cols_by_group").expect("sequence_cols_by_group").split('|') {
        let (g, c) = entry.split_once(':').expect("group:col");
        let i = seq_cols.iter().position(|x| x == c).unwrap_or_else(|| panic!("unknown seq col {c}"));
        seq_of_group.insert(g.to_string(), i);
    }
    let weights = map
        .get("weights")
        .expect("weights")
        .split('|')
        .map(|e| {
            let (g, w) = e.split_once(':').expect("group:weight");
            (g.to_string(), w.parse().expect("weight"))
        })
        .collect();
    let curve_blend = map
        .get("structure_curve_blend")
        .map(|v| {
            let mut it = v.split('|');
            (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
        })
        .unwrap_or((0.6, 0.4));
    let seq_weights = map
        .get("combined_seq_weights")
        .map(|v| {
            let n: Vec<f64> = v.split('|').map(|x| x.parse().unwrap()).collect();
            [n[0], n[1], n[2], n[3]]
        })
        .unwrap_or([0.35, 0.25, 0.2, 0.2]);
    let cfg = Cfg {
        groups,
        seq_of_group,
        weights,
        blend_scalar_sequence: map.get("blend_scalar_sequence").and_then(|v| v.parse().ok()).unwrap_or(0.5),
        curve_blend,
        seq_weights,
        ngram_n: map.get("ngram_n").and_then(|v| v.parse().ok()).unwrap_or(2),
        limit: map.get("limit").and_then(|v| v.parse().ok()).unwrap_or(20),
    };

    let raw = fs::read_to_string(format!("{dir}/tracks.tsv")).expect("tracks.tsv");
    let mut lines = raw.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap_or_else(|| panic!("missing column {name}"));
    let scalar_at: Vec<usize> = scalar_cols.iter().map(|c| col(c)).collect();
    let seq_at: Vec<usize> = seq_cols.iter().map(|c| col(c)).collect();
    let (curve_at, vocal_at, emb_at) = (col("energy_curve"), col("has_vocal"), col("embedding"));

    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut tracks = Vec::new();
    for line in lines.filter(|l| !l.is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        debug_assert_eq!(f.len(), header.len(), "a token contained a tab");
        let seqs = seq_at
            .iter()
            .map(|&i| {
                if f[i].is_empty() {
                    Vec::new()
                } else {
                    f[i]
                        .split(SEP)
                        .map(|tok| {
                            let next = vocab.len() as u32;
                            *vocab.entry(tok.to_string()).or_insert(next)
                        })
                        .collect()
                }
            })
            .collect();
        tracks.push(Track {
            id: f[0].to_string(),
            scalars: scalar_at.iter().map(|&i| parse_opt(f[i])).collect(),
            seqs,
            curve: if f[curve_at].is_empty() { Vec::new() } else { f[curve_at].split(SEP).filter_map(parse_opt).collect() },
            has_vocal: f[vocal_at] == "1",
            emb: if f[emb_at].is_empty() { Vec::new() } else { f[emb_at].split(SEP).filter_map(parse_opt).collect() },
        });
    }

    let ncols = scalar_cols.len();
    let mut sorted = vec![Vec::new(); ncols];
    for t in &tracks {
        for (i, v) in t.scalars.iter().enumerate() {
            if let Some(v) = v {
                sorted[i].push(*v);
            }
        }
    }
    for col in sorted.iter_mut() {
        col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }
    Space { tracks, sorted, cfg }
}

fn count_map(tokens: &[u32], n: usize) -> BTreeMap<Vec<u32>, usize> {
    let mut out = BTreeMap::new();
    if tokens.len() >= n {
        for g in tokens.windows(n) {
            *out.entry(g.to_vec()).or_insert(0) += 1;
        }
    }
    out
}

fn counter_overlap(a: &BTreeMap<Vec<u32>, usize>, b: &BTreeMap<Vec<u32>, usize>) -> (usize, usize) {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (k, &va) in a {
        let vb = b.get(k).copied().unwrap_or(0);
        inter += va.min(vb);
        union += va.max(vb);
    }
    for (k, vb) in b {
        if !a.contains_key(k) {
            union += vb;
        }
    }
    (inter, union)
}

fn counter_ratio(a: &BTreeMap<Vec<u32>, usize>, b: &BTreeMap<Vec<u32>, usize>, both_empty: f64) -> f64 {
    if a.is_empty() && b.is_empty() {
        return both_empty;
    }
    let (inter, union) = counter_overlap(a, b);
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Levenshtein similarity on interned token sequences: the textbook row DP.
fn levenshtein_similarity(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let (m, n) = (a.len(), b.len());
    let mut prev: Vec<usize> = (0..=n).collect();
    for i in 1..=m {
        let mut cur = vec![i; n + 1];
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    1.0 - prev[n] as f64 / m.max(n) as f64
}

fn jaccard(a: &[u32], b: &[u32]) -> f64 {
    let sa: HashSet<u32> = a.iter().copied().collect();
    let sb: HashSet<u32> = b.iter().copied().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn transitions(a: &[u32]) -> BTreeMap<Vec<u32>, usize> {
    let mut out = BTreeMap::new();
    for pair in a.windows(2) {
        *out.entry(pair.to_vec()).or_insert(0) += 1;
    }
    out
}

fn combined_sequence_similarity(a: &[u32], b: &[u32], cfg: &Cfg) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let w = &cfg.seq_weights;
    let n = cfg.ngram_n;
    let ng = counter_ratio(&count_map(a, n), &count_map(b, n), 1.0);
    let tr = counter_ratio(&transitions(a), &transitions(b), 1.0);
    w[0] * levenshtein_similarity(a, b) + w[1] * ng + w[2] * jaccard(a, b) + w[3] * tr
}

fn clip01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    let mut sab = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        saa += da * da;
        sbb += db * db;
        sab += da * db;
    }
    if saa > 0.0 && sbb > 0.0 {
        sab / (saa.sqrt() * sbb.sqrt())
    } else {
        0.0
    }
}

impl Space {
    fn percentile(&self, ti: usize, col: usize) -> Option<f64> {
        let v = self.tracks[ti].scalars[col]?;
        let arr = &self.sorted[col];
        if arr.is_empty() {
            return None;
        }
        let i = arr.partition_point(|x| *x <= v);
        Some(i as f64 / arr.len() as f64)
    }

    fn structure_similarity(&self, a: usize, b: usize, seq_col: Option<usize>) -> Option<f64> {
        let mut seq_sim = None;
        if let Some(sc) = seq_col {
            let (sa, sb) = (&self.tracks[a].seqs[sc], &self.tracks[b].seqs[sc]);
            if sa.len() >= 2 && sb.len() >= 2 {
                seq_sim = Some(combined_sequence_similarity(sa, sb, &self.cfg));
            }
        }
        let (ca, cb) = (&self.tracks[a].curve, &self.tracks[b].curve);
        if ca.len() == cb.len() && ca.len() > 1 && ca.iter().any(|x| *x != 0.0) && cb.iter().any(|x| *x != 0.0) {
            let curve = clip01((pearson(ca, cb) + 1.0) / 2.0);
            return Some(match seq_sim {
                None => curve,
                Some(s) => clip01(self.cfg.curve_blend.0 * curve + self.cfg.curve_blend.1 * s),
            });
        }
        seq_sim
    }

    fn group_similarity(&self, a: usize, b: usize, group: &str) -> Option<f64> {
        if group == "structure" {
            return self.structure_similarity(a, b, self.cfg.seq_of_group.get(group).copied());
        }
        let cols = self.cfg.groups.get(group)?;
        if group == "vocal" && !(self.tracks[a].has_vocal && self.tracks[b].has_vocal) {
            return None;
        }
        let mut sum = 0.0;
        let mut n = 0usize;
        for &c in cols {
            if let (Some(pa), Some(pb)) = (self.percentile(a, c), self.percentile(b, c)) {
                sum += (pa - pb).abs();
                n += 1;
            }
        }
        let scalar = if n > 0 { Some(clip01(1.0 - sum / n as f64)) } else { None };
        if let Some(&sc) = self.cfg.seq_of_group.get(group) {
            let (sa, sb) = (&self.tracks[a].seqs[sc], &self.tracks[b].seqs[sc]);
            if sa.len() >= 2 && sb.len() >= 2 {
                let seq = combined_sequence_similarity(sa, sb, &self.cfg);
                return Some(match scalar {
                    None => seq,
                    Some(s) => clip01(self.cfg.blend_scalar_sequence * s + (1.0 - self.cfg.blend_scalar_sequence) * seq),
                });
            }
        }
        scalar
    }

    fn emb_sim(&self, a: usize, b: usize) -> Option<f64> {
        let (va, vb) = (&self.tracks[a].emb, &self.tracks[b].emb);
        if va.is_empty() || va.len() != vb.len() {
            return None;
        }
        // FAISS stores float32 and accumulates in float32; mirror that so the cosine
        // agrees with Python's rather than differing in the 7th decimal.
        let mut dot: f32 = 0.0;
        for i in 0..va.len() {
            dot += (va[i] as f32) * (vb[i] as f32);
        }
        Some(clip01((f64::from(dot) + 1.0) / 2.0))
    }

    fn score_pair(&self, a: usize, b: usize, emb: Option<f64>) -> (f64, BTreeMap<String, f64>) {
        let mut detail = BTreeMap::new();
        let mut total = 0.0;
        let mut weight_sum = 0.0;
        for (group, weight) in self.cfg.weights.iter() {
            let sim = if group == "embedding" {
                emb
            } else {
                self.group_similarity(a, b, group)
            };
            if let Some(s) = sim {
                detail.insert(group.clone(), s);
                total += weight * s;
                weight_sum += weight;
            }
        }
        let score = if weight_sum > 0.0 { total / weight_sum } else { 0.0 };
        (score, detail)
    }

    /// song -> song: rank every other track, mirroring the Python flow.
    fn similar(&self, seed: usize) -> Vec<(usize, f64)> {
        let mut cands: Vec<(usize, f64)> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != seed)
            .map(|(i, _)| {
                let s = self.emb_sim(seed, i).unwrap_or(0.0);
                (i, s)
            })
            .collect();
        // FAISS hands Python candidates in descending similarity; keep that order so
        // equal scores break the same way (sort_by is stable).
        cands.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut scored: Vec<(usize, f64)> = cands
            .iter()
            .map(|(i, e)| (*i, self.score_pair(seed, *i, Some(*e)).0))
            .collect();
        scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.cfg.limit);
        scored
    }
}

fn expected(dir: &str) -> BTreeMap<String, Vec<(usize, String, f64)>> {
    let mut out: BTreeMap<String, Vec<(usize, String, f64)>> = BTreeMap::new();
    let raw = match fs::read_to_string(format!("{dir}/expected_top20.tsv")) {
        Ok(s) => s,
        Err(_) => return out,
    };
    for line in raw.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            continue;
        }
        out.entry(f[0].to_string()).or_default().push((f[1].parse().unwrap(), f[2].to_string(), f[3].parse().unwrap()));
    }
    out
}

/// Diff the port's rankings against Python's exported top-N.
/// Returns (order mismatches, max |score delta|, comparisons actually made).
fn parity(space: &Space, exp: &BTreeMap<String, Vec<(usize, String, f64)>>, report: bool) -> (usize, f64, usize) {
    let mut order_bad = 0usize;
    let mut max_delta = 0f64;
    let mut compared = 0usize;
    for (seed_id, rows) in exp.iter() {
        let si = match space.tracks.iter().position(|t| &t.id == seed_id) {
            Some(i) => i,
            None => {
                println!("seed {seed_id} missing from export");
                order_bad += 1;
                continue;
            }
        };
        let got = space.similar(si);
        if got.len() != rows.len() {
            println!("{seed_id}: length {} != {}", got.len(), rows.len());
            order_bad += 1;
            continue;
        }
        for (rank, (ci, cscore)) in got.iter().enumerate() {
            let (erank, eid, escore) = &rows[rank];
            if erank != &rank {
                continue;
            }
            compared += 1;
            if &space.tracks[*ci].id != eid {
                if report && order_bad < 5 {
                    println!("{seed_id} rank {rank}: rust={} python={}", space.tracks[*ci].id, eid);
                }
                order_bad += 1;
            }
            max_delta = max_delta.max((cscore.round_to_4dp() - escore).abs());
        }
    }
    (order_bad, max_delta, compared)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or_else(|| "data/portability".to_string());
    let verify = args.iter().any(|a| a == "--verify");
    let t_load = Instant::now();
    let space = load(&dir);
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    println!("loaded {} tracks, {} scalar columns, {} weights in {:.1} ms", space.tracks.len(), space.sorted.len(), space.cfg.weights.len(), load_ms);

    let mut t0 = Instant::now();
    let mut per_query = Vec::new();
    for round in 0..2 {
        // round 0 warms caches and branch predictors; report round 1
        if round == 1 {
            per_query.clear();
            t0 = Instant::now();
        }
        for seed in 0..space.tracks.len() {
            let s = Instant::now();
            let _ = space.similar(seed);
            per_query.push(s.elapsed().as_secs_f64() * 1000.0);
        }
    }
    per_query.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "all {} seeds: median {:.2} ms/query, max {:.2} ms (host x86 release build)",
        per_query.len(),
        per_query[per_query.len() / 2],
        per_query[per_query.len() - 1]
    );
    println!("total {:.1} ms for the whole library", t0.elapsed().as_secs_f64() * 1000.0);

    let exp = expected(&dir);
    if exp.is_empty() {
        println!("no expected_top20.tsv -> parity not checked");
        return;
    }
    let (order_bad, max_delta, compared) = parity(&space, &exp, true);
    println!("parity vs Python over {} seeds / {} ranked slots: order mismatches = {order_bad}, max |score delta| = {max_delta:.2e}", exp.len(), compared);
    if verify {
        // Control: a zero-mismatch run is only meaningful if a deliberate weight
        // break is caught by the same comparison. Without this a misparsed manifest
        // could pass by comparing nothing.
        let mut broken = load(&dir);
        broken.cfg.weights.iter_mut().find(|(g, _)| g != "embedding").unwrap().1 *= 2.0;
        let (c_bad, c_delta, c_compared) = parity(&broken, &exp, false);
        println!("control (one weight doubled): order mismatches = {c_bad}, max |score delta| = {c_delta:.2e}");
        if c_bad == 0 || c_compared < compared {
            println!("CONTROL FAILED: parity diff cannot see a weight change -> not trustworthy");
            exit(1);
        }
        if order_bad != 0 || max_delta > 1e-4 {
            println!("FAIL");
            exit(1);
        }
    }
    println!("OK");
}

trait Round4 {
    fn round_to_4dp(self) -> f64;
}
impl Round4 for f64 {
    fn round_to_4dp(self) -> f64 {
        (self * 10000.0).round() / 10000.0
    }
}
