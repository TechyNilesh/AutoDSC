use pyo3::prelude::*;
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

#[pyclass]
pub struct StreamKMeans {
    chunk_size: usize,
    n_clusters: usize,
    halflife: f64,
    mu: f64,
    sigma: f64,
    p: f64,
    seed: Option<u64>,
    
    time_stamp: usize,
    kmeans: KMeansInternal,
    temp_chunk: HashMap<usize, Vec<f64>>,
    centers: HashMap<usize, Vec<f64>>,
    cluster_scatter: HashMap<usize, f64>,
    n_features: usize,
    initialized: bool,
}

#[pymethods]
impl StreamKMeans {
    #[new]
    #[pyo3(signature = (chunk_size=200, n_clusters=8, halflife=0.5, mu=0.0, sigma=1.0, p=2.0, seed=None))]
    fn new(
        chunk_size: usize,
        n_clusters: usize,
        halflife: f64,
        mu: f64,
        sigma: f64,
        p: f64,
        seed: Option<u64>,
    ) -> Self {
        let kmeans = KMeansInternal::new(n_clusters, halflife, mu, sigma, p, seed);
        
        Self {
            chunk_size,
            n_clusters,
            halflife,
            mu,
            sigma,
            p,
            seed,
            time_stamp: 0,
            kmeans,
            temp_chunk: HashMap::new(),
            centers: HashMap::new(),
            cluster_scatter: HashMap::new(),
            n_features: 0,
            initialized: false,
        }
    }
    
    #[pyo3(signature = (x, w=None))]
    fn learn_one(&mut self, x: Vec<f64>, w: Option<f64>) {
        if !self.initialized {
            self.n_features = x.len();
            self.kmeans.initialize_centers(self.n_features);
            self.initialized = true;
        }

        // Update EMA scatter for closest cluster
        if !self.centers.is_empty() {
            let closest = self.kmeans.predict_one(x.clone());
            if let Some(center) = self.centers.get(&closest) {
                let dist = euclidean_distance(&x, center);
                let prev = self.cluster_scatter.get(&closest).copied().unwrap_or(dist);
                self.cluster_scatter.insert(closest, prev + self.halflife * (dist - prev));
            }
        }

        self.time_stamp += 1;
        let index = self.time_stamp % self.chunk_size;
        
        if index == 0 {
            self.temp_chunk.insert(self.chunk_size - 1, x);
        } else if index == 1 {
            self.temp_chunk.clear();
            self.temp_chunk.insert(0, x);
        } else {
            self.temp_chunk.insert(index - 1, x);
        }
        
        // When chunk is full, process it
        if index == 0 {
            let mut kmeans_i = KMeansInternal::new(
                self.n_clusters,
                self.halflife,
                self.mu,
                self.sigma,
                self.p,
                self.seed,
            );
            kmeans_i.initialize_centers(self.n_features);

            // Collect chunk points in stable index order. temp_chunk is a HashMap,
            // so .values() iteration order is not deterministic across runs; sorting
            // by the stored index restores the original arrival order.
            let mut chunk_indices: Vec<usize> = self.temp_chunk.keys().cloned().collect();
            chunk_indices.sort_unstable();
            let chunk_points: Vec<Vec<f64>> = chunk_indices
                .iter()
                .map(|i| self.temp_chunk[i].clone())
                .collect();

            // Initialize chunk k-means from evenly spaced chunk points
            kmeans_i.initialize_centers_from_points(&chunk_points);

            // Learn from all points in the temporary chunk (arrival order)
            for point in &chunk_points {
                kmeans_i.learn_one(point.clone());
            }

            // Learn from the centers of the chunk k-means
            for center in kmeans_i.get_centers() {
                self.kmeans.learn_one(center);
            }
        }
        
        self.centers = self.kmeans.get_centers_map();
    }
    
    #[pyo3(signature = (x, w=None))]
    fn predict_one(&self, x: Vec<f64>, w: Option<f64>) -> usize {
        if self.centers.is_empty() {
            return 0;
        }
        
        let mut min_distance = f64::INFINITY;
        let mut closest_cluster = 0;

        // Iterate cluster ids in fixed order so ties break deterministically
        // (HashMap iteration order is not seeded and varies across runs).
        for cluster_id in 0..self.n_clusters {
            if let Some(center) = self.centers.get(&cluster_id) {
                let distance = euclidean_distance(center, &x);
                if distance < min_distance {
                    min_distance = distance;
                    closest_cluster = cluster_id;
                }
            }
        }

        closest_cluster
    }

    fn get_centers(&self) -> Vec<Vec<f64>> {
        let mut centers = Vec::new();
        for i in 0..self.n_clusters {
            if let Some(center) = self.centers.get(&i) {
                centers.push(center.clone());
            } else {
                centers.push(vec![0.0; self.n_features]);
            }
        }
        centers
    }

    fn get_n_clusters(&self) -> usize {
        self.n_clusters
    }
    
    fn get_centroids(&self) -> Vec<(Vec<f64>, f64, f64)> {
        (0..self.n_clusters).filter_map(|i| {
            self.centers.get(&i).map(|c| {
                let s = self.cluster_scatter.get(&i).copied().unwrap_or(0.0);
                (c.clone(), 1.0f64, s)
            })
        }).collect()
    }

    fn get_chunk_size(&self) -> usize {
        self.chunk_size
    }
    
    fn get_time_stamp(&self) -> usize {
        self.time_stamp
    }
}

// Internal KMeans implementation for chunk processing
struct KMeansInternal {
    n_clusters: usize,
    halflife: f64,
    mu: f64,
    sigma: f64,
    p: f64,
    seed: Option<u64>,
    centers: HashMap<usize, Vec<f64>>,
    n_features: usize,
    initialized: bool,
    rng: StdRng,
}

impl KMeansInternal {
    fn new(
        n_clusters: usize,
        halflife: f64,
        mu: f64,
        sigma: f64,
        p: f64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            n_clusters,
            halflife,
            mu,
            sigma,
            p,
            seed,
            centers: HashMap::new(),
            n_features: 0,
            initialized: false,
            rng,
        }
    }

    fn initialize_centers(&mut self, n_features: usize) {
        self.n_features = n_features;
        // Centers will be populated from the first n_clusters data points
        self.centers.clear();
        self.initialized = true;
    }

    fn initialize_centers_from_points(&mut self, points: &[Vec<f64>]) {
        self.centers.clear();
        let n = points.len();
        for i in 0..self.n_clusters {
            let idx = (i * n) / self.n_clusters;
            self.centers.insert(i, points[idx].clone());
        }
    }

    fn learn_one(&mut self, x: Vec<f64>) {
        if !self.initialized {
            self.initialize_centers(x.len());
        }

        // Buffer first n_clusters points as initial centers
        if self.centers.len() < self.n_clusters {
            let idx = self.centers.len();
            self.centers.insert(idx, x);
            return;
        }

        // Find the cluster with the closest center
        let closest = self.predict_one(x.clone());

        // Move the cluster's center
        if let Some(center) = self.centers.get_mut(&closest) {
            for i in 0..self.n_features.min(x.len()) {
                center[i] += self.halflife * (x[i] - center[i]);
            }
        }
    }
    
    fn predict_one(&self, x: Vec<f64>) -> usize {
        if !self.initialized {
            return 0;
        }
        
        let mut min_distance = f64::INFINITY;
        let mut closest_cluster = 0;

        // Deterministic tie-breaking: visit cluster ids in order, not HashMap order.
        for cluster_id in 0..self.n_clusters {
            if let Some(center) = self.centers.get(&cluster_id) {
                let distance = self.get_distance(center, &x);
                if distance < min_distance {
                    min_distance = distance;
                    closest_cluster = cluster_id;
                }
            }
        }

        closest_cluster
    }
    
    fn get_centers(&self) -> Vec<Vec<f64>> {
        let mut centers = Vec::new();
        for i in 0..self.n_clusters {
            if let Some(center) = self.centers.get(&i) {
                centers.push(center.clone());
            } else {
                centers.push(vec![0.0; self.n_features]);
            }
        }
        centers
    }
    
    fn get_centers_map(&self) -> HashMap<usize, Vec<f64>> {
        self.centers.clone()
    }
    
    fn get_distance(&self, center: &[f64], x: &[f64]) -> f64 {
        let mut distance = 0.0;
        
        for i in 0..self.n_features.min(x.len()) {
            let diff = (center[i] - x[i]).abs();
            distance += diff.powf(self.p);
        }
        
        distance.powf(1.0 / self.p)
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[pymodule]
fn streamkmeans_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StreamKMeans>()?;
    Ok(())
}
