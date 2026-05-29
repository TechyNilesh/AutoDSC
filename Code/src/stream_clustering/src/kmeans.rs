use pyo3::prelude::*;
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

#[pyclass]
pub struct KMeans {
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
    init_buffer: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, halflife=0.5, mu=0.0, sigma=1.0, p=2.0, seed=None))]
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
            init_buffer: None,
        }
    }
    
    fn learn_predict_one(&mut self, x: Vec<f64>) -> usize {
        if !self.initialized {
            self.initialize_centers(x.len());
        }

        // Buffer initial points to get diverse center initialization
        if let Some(ref mut buffer) = self.init_buffer {
            buffer.push(x.clone());
            if buffer.len() >= self.n_clusters * 10 {
                self.initialize_centers_from_buffer();
            } else {
                return 0;
            }
        }

        // Find the cluster with the closest center
        let closest = self.predict_one(x.clone());

        // Move the cluster's center
        if let Some(center) = self.centers.get_mut(&closest) {
            for i in 0..self.n_features.min(x.len()) {
                center[i] += self.halflife * (x[i] - center[i]);
            }
        }

        closest
    }
    
    fn learn_one(&mut self, x: Vec<f64>) {
        self.learn_predict_one(x);
    }
    
    fn predict_one(&self, x: Vec<f64>) -> usize {
        if !self.initialized {
            return 0;
        }
        
        let mut min_distance = f64::INFINITY;
        let mut closest_cluster = 0;

        // Iterate cluster ids in fixed order so ties break deterministically
        // (HashMap iteration order is not seeded and varies across runs).
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
    
    fn get_n_clusters(&self) -> usize {
        self.n_clusters
    }
    
    fn set_halflife(&mut self, halflife: f64) {
        self.halflife = halflife;
    }
    
    fn get_halflife(&self) -> f64 {
        self.halflife
    }
}

impl KMeans {
    fn initialize_centers(&mut self, n_features: usize) {
        self.n_features = n_features;
        self.centers.clear();
        // Defer actual initialization until we have enough data points
        // to sample diverse centers from the stream
        self.init_buffer = Some(Vec::new());
        self.initialized = true;
    }

    fn initialize_centers_from_buffer(&mut self) {
        if let Some(ref buffer) = self.init_buffer {
            let n = buffer.len();
            if n == 0 { return; }
            // Pick evenly spaced points from the buffer for diverse initialization
            for i in 0..self.n_clusters {
                let idx = (i * n) / self.n_clusters;
                self.centers.insert(i, buffer[idx].clone());
            }
        }
        self.init_buffer = None;
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

#[pymodule]
fn kmeans_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<KMeans>()?;
    Ok(())
}
