use pyo3::prelude::*;
use rand::prelude::*;

// Simple incremental variance calculator similar to River's stats.Var
#[derive(Clone, Debug)]
struct IncrementalVar {
    n: f64,
    mean: f64,
    m2: f64,  // sum of squares of deviations from mean
}

impl IncrementalVar {
    fn new() -> Self {
        Self {
            n: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn update(&mut self, value: f64, weight: f64) {
        self.n += weight;

        if self.n == 0.0 {
            return;
        }

        let delta = value - self.mean;
        self.mean += delta * weight / self.n;
        let delta2 = value - self.mean;
        self.m2 += weight * delta * delta2;
    }

    fn get_mean(&self) -> f64 {
        self.mean
    }

    fn get_variance(&self) -> f64 {
        if self.n <= 1.0 {
            0.0
        } else {
            self.m2 / self.n
        }
    }

    fn get_count(&self) -> f64 {
        self.n
    }
}

impl std::ops::AddAssign<&IncrementalVar> for IncrementalVar {
    fn add_assign(&mut self, other: &IncrementalVar) {
        if other.n == 0.0 {
            return;
        }
        if self.n == 0.0 {
            *self = other.clone();
            return;
        }

        let combined_n = self.n + other.n;
        let delta = other.mean - self.mean;
        let combined_mean = (self.n * self.mean + other.n * other.mean) / combined_n;
        let combined_m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / combined_n;

        self.n = combined_n;
        self.mean = combined_mean;
        self.m2 = combined_m2;
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CluStreamMicroCluster {
    var_x: Vec<IncrementalVar>,
    var_time: IncrementalVar,
    n_features: usize,
    cached_center: Vec<f64>,
}

#[pymethods]
impl CluStreamMicroCluster {
    #[new]
    fn new(x: Vec<f64>, w: f64, timestamp: i64) -> Self {
        let n_features = x.len();
        let mut var_x = Vec::with_capacity(n_features);

        for &val in &x {
            let mut var = IncrementalVar::new();
            var.update(val, w);
            var_x.push(var);
        }

        let mut var_time = IncrementalVar::new();
        var_time.update(timestamp as f64, w);

        let cached_center = x.clone();

        Self {
            var_x,
            var_time,
            n_features,
            cached_center,
        }
    }

    fn center(&self) -> Vec<f64> {
        self.cached_center.clone()
    }

    fn radius(&self, r_factor: f64) -> f64 {
        if self.weight() == 1.0 {
            return 0.0;
        }
        self.deviation() * r_factor
    }

    fn deviation(&self) -> f64 {
        let mut dev_sum = 0.0;
        for var in &self.var_x {
            dev_sum += var.get_variance().sqrt();
        }
        if self.n_features > 0 {
            dev_sum / self.n_features as f64
        } else {
            f64::INFINITY
        }
    }

    fn weight(&self) -> f64 {
        self.var_time.get_count()
    }

    fn insert(&mut self, x: Vec<f64>, w: f64, timestamp: i64) {
        self.var_time.update(timestamp as f64, w);
        for (i, &val) in x.iter().enumerate() {
            if i < self.var_x.len() {
                self.var_x[i].update(val, w);
                self.cached_center[i] = self.var_x[i].get_mean();
            }
        }
    }

    fn relevance_stamp(&self, max_mc: i64) -> f64 {
        let mu_time = self.var_time.get_mean();

        if self.weight() < 2.0 * max_mc as f64 {
            return mu_time;
        }

        let sigma_time = self.var_time.get_variance().sqrt();
        mu_time + sigma_time * self.quantile(max_mc as f64 / (2.0 * self.weight()))
    }

    fn quantile(&self, z: f64) -> f64 {
        (2.0_f64).sqrt() * self.inverse_error(2.0 * z - 1.0)
    }

    fn inverse_error(&self, x: f64) -> f64 {
        let z = (std::f64::consts::PI).sqrt() * x;
        let mut res = x / 2.0;
        let z2 = z * z;
        let mut zprod = z2 * z;

        res += (1.0 / 24.0) * zprod;
        zprod *= z2;
        res += (7.0 / 960.0) * zprod;
        zprod *= z2;
        res += (127.0 * zprod) / 80640.0;
        zprod *= z2;
        res += (4369.0 * zprod) / 11612160.0;
        zprod *= z2;
        res += (34807.0 * zprod) / 364953600.0;
        zprod *= z2;
        res += (20036983.0 * zprod) / 797058662400.0;

        res
    }

    fn merge_with(&mut self, other: &CluStreamMicroCluster) {
        self.var_time += &other.var_time;
        for (i, other_var) in other.var_x.iter().enumerate() {
            if i < self.var_x.len() {
                self.var_x[i] += other_var;
                self.cached_center[i] = self.var_x[i].get_mean();
            }
        }
    }
}

#[pyclass]
pub struct CluStream {
    n_macro_clusters: usize,
    max_micro_clusters: usize,
    micro_cluster_r_factor: f64,
    time_window: i64,
    time_gap: i64,
    micro_clusters: Vec<Option<CluStreamMicroCluster>>,
    centers: Vec<Vec<f64>>,
    macro_cluster_weights: Vec<f64>,
    macro_cluster_scatter: Vec<f64>,
    timestamp: i64,
    initialized: bool,
    mc_centers: Vec<Vec<f64>>,
    rng: StdRng,
}

#[pymethods]
impl CluStream {
    #[new]
    #[pyo3(signature = (n_macro_clusters=5, max_micro_clusters=100, micro_cluster_r_factor=2.0, time_window=1000, time_gap=100, seed=None))]
    fn new(
        n_macro_clusters: usize,
        max_micro_clusters: usize,
        micro_cluster_r_factor: f64,
        time_window: i64,
        time_gap: i64,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            n_macro_clusters,
            max_micro_clusters,
            micro_cluster_r_factor,
            time_window,
            time_gap,
            micro_clusters: vec![None; max_micro_clusters],
            centers: Vec::new(),
            macro_cluster_weights: Vec::new(),
            macro_cluster_scatter: Vec::new(),
            timestamp: -1,
            initialized: false,
            mc_centers: Vec::new(),
            rng,
        }
    }
    
    #[pyo3(signature = (x, w=1.0))]
    fn learn_one(&mut self, x: Vec<f64>, w: f64) {
        self.timestamp += 1;

        if !self.initialized {
            let mc = CluStreamMicroCluster::new(x, w, self.max_micro_clusters as i64 - 1);

            // Find first available slot
            for i in 0..self.max_micro_clusters {
                if self.micro_clusters[i].is_none() {
                    self.micro_clusters[i] = Some(mc);
                    break;
                }
            }

            let occupied_count = self.micro_clusters.iter().filter(|mc| mc.is_some()).count();
            if occupied_count == self.max_micro_clusters {
                self.initialized = true;
            }
            return;
        }

        // Determine the closest micro-cluster with respect to the new point instance
        let (closest_id, closest_dist) = self.get_closest_mc(&x);

        // Check whether the new instance fits into the closest micro-cluster
        let radius = if let Some(ref closest_mc) = self.micro_clusters[closest_id] {
            if closest_mc.weight() == 1.0 {
                let center = closest_mc.center();
                self.get_min_distance_to_other_mc(closest_id, &center)
            } else {
                closest_mc.radius(self.micro_cluster_r_factor)
            }
        } else {
            f64::INFINITY
        };

        if closest_dist < radius {
            if let Some(ref mut mc) = self.micro_clusters[closest_id] {
                mc.insert(x, w, self.timestamp);
            }
            return;
        }

        // If the new point does not fit in the micro-cluster, micro-clusters
        // whose relevance stamps are less than the threshold are deleted.
        // Otherwise, closest micro-clusters are merged with each other.
        self.maintain_micro_clusters(x, w);

        // Apply incremental K-Means on micro-clusters after each time_gap
        if self.timestamp % self.time_gap == self.time_gap - 1 {
            // Micro-cluster centers will only be saved when the calculation of macro-cluster centers
            // is required, in order not to take up memory and time unnecessarily
            self.mc_centers = self.micro_clusters.iter()
                .filter_map(|mc_opt| mc_opt.as_ref())
                .map(|mc| mc.center())
                .collect();

            self.update_macro_clusters();
        }
    }
    
    fn predict_one(&self, x: Vec<f64>) -> usize {
        if self.mc_centers.is_empty() || self.centers.is_empty() {
            return 0;
        }

        // Find the closest micro-cluster center directly from mc_centers
        // (cannot use get_closest_mc index because mc_centers skips None entries)
        let mut closest_dist = f64::INFINITY;
        let mut closest_mc_center = &self.mc_centers[0];
        for center in &self.mc_centers {
            let dist = euclidean_distance(&x, center);
            if dist < closest_dist {
                closest_dist = dist;
                closest_mc_center = center;
            }
        }

        // Map the closest micro-cluster to a macro-cluster
        self.get_closest_macro_cluster(closest_mc_center)
    }
    
    fn get_centers(&self) -> Vec<Vec<f64>> {
        self.centers.clone()
    }

    fn get_centroids(&self) -> Vec<(Vec<f64>, f64, f64)> {
        self.centers.iter().enumerate().map(|(i, c)| {
            let w = self.macro_cluster_weights.get(i).copied().unwrap_or(1.0);
            let s = self.macro_cluster_scatter.get(i).copied().unwrap_or(0.0);
            (c.clone(), w.max(1.0), s)
        }).collect()
    }
}

impl CluStream {
    fn get_closest_mc(&self, x: &[f64]) -> (usize, f64) {
        let mut closest_dist_sq = f64::INFINITY;
        let mut closest_idx = 0;

        for (mc_idx, mc_opt) in self.micro_clusters.iter().enumerate() {
            if let Some(mc) = mc_opt {
                let dist_sq = euclidean_distance_sq(x, &mc.cached_center);
                if dist_sq < closest_dist_sq {
                    closest_dist_sq = dist_sq;
                    closest_idx = mc_idx;
                }
            }
        }

        (closest_idx, closest_dist_sq.sqrt())
    }
    
    fn get_min_distance_to_other_mc(&self, exclude_id: usize, center: &[f64]) -> f64 {
        let mut min_dist_sq = f64::INFINITY;

        for (mc_id, mc_opt) in self.micro_clusters.iter().enumerate() {
            if mc_id == exclude_id { continue; }
            if let Some(mc) = mc_opt {
                let dist_sq = euclidean_distance_sq(center, &mc.cached_center);
                if dist_sq < min_dist_sq { min_dist_sq = dist_sq; }
            }
        }

        min_dist_sq.sqrt()
    }
    
    fn maintain_micro_clusters(&mut self, x: Vec<f64>, w: f64) {
        // Calculate the threshold to delete old micro-clusters
        let threshold = self.timestamp - self.time_window;

        // Delete old micro-cluster if its relevance stamp is smaller than the threshold
        let mut del_id = None;
        for (i, mc_opt) in self.micro_clusters.iter().enumerate() {
            if let Some(mc) = mc_opt {
                if mc.relevance_stamp(self.max_micro_clusters as i64) < threshold as f64 {
                    del_id = Some(i);
                    break;
                }
            }
        }

        if let Some(id) = del_id {
            let new_mc = CluStreamMicroCluster::new(x, w, self.timestamp);
            self.micro_clusters[id] = Some(new_mc);
            return;
        }

        // Merge the two closest micro-clusters
        let (closest_a, closest_b) = self.find_closest_pair();

        if let (Some(mc_a), Some(mc_b)) = (
            self.micro_clusters[closest_a].clone(),
            self.micro_clusters[closest_b].clone(),
        ) {
            let mut merged = mc_a;
            merged.merge_with(&mc_b);

            let new_mc = CluStreamMicroCluster::new(x, w, self.timestamp);
            self.micro_clusters[closest_a] = Some(merged);
            self.micro_clusters[closest_b] = Some(new_mc);
        }
    }
    
    fn find_closest_pair(&self) -> (usize, usize) {
        let mut min_dist_sq = f64::INFINITY;
        let mut closest_a = 0;
        let mut closest_b = 0;

        for i in 0..self.micro_clusters.len() {
            if let Some(mc_a) = &self.micro_clusters[i] {
                for j in (i + 1)..self.micro_clusters.len() {
                    if let Some(mc_b) = &self.micro_clusters[j] {
                        let dist_sq = euclidean_distance_sq(&mc_a.cached_center, &mc_b.cached_center);
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                            closest_a = i;
                            closest_b = j;
                        }
                    }
                }
            }
        }

        (closest_a, closest_b)
    }
    
    fn update_macro_clusters(&mut self) {
        if self.mc_centers.is_empty() {
            return;
        }

        // Warm-start k-means with previous centers to keep cluster IDs stable
        let clustered = if self.centers.is_empty() || self.centers.len() != self.n_macro_clusters {
            kmeans(&self.mc_centers, self.n_macro_clusters, &mut self.rng)
        } else {
            kmeans_with_init(&self.mc_centers, &self.centers)
        };
        self.centers = clustered;

        // Compute weight and scatter of each macro-cluster from assigned micro-clusters
        let active_mcs: Vec<&CluStreamMicroCluster> = self.micro_clusters.iter()
            .filter_map(|mc_opt| mc_opt.as_ref())
            .collect();
        let mut weights = vec![0.0f64; self.centers.len()];
        let mut scatter = vec![0.0f64; self.centers.len()];
        let mut counts  = vec![0usize;  self.centers.len()];
        for (mc_center, mc) in self.mc_centers.iter().zip(active_mcs.iter()) {
            let closest = self.get_closest_macro_cluster(mc_center);
            weights[closest] += mc.weight();
            scatter[closest] += mc.deviation();
            counts[closest]  += 1;
        }
        for k in 0..self.centers.len() {
            if counts[k] > 0 {
                scatter[k] /= counts[k] as f64;
            }
        }
        self.macro_cluster_weights = weights;
        self.macro_cluster_scatter = scatter;
    }
    
    fn get_closest_macro_cluster(&self, point: &[f64]) -> usize {
        let mut closest_dist = f64::INFINITY;
        let mut closest_idx = 0;
        
        for (idx, center) in self.centers.iter().enumerate() {
            let dist = euclidean_distance(point, center);
            if dist < closest_dist {
                closest_dist = dist;
                closest_idx = idx;
            }
        }
        
        closest_idx
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    euclidean_distance_sq(a, b).sqrt()
}

fn euclidean_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn kmeans(points: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    if points.is_empty() {
        return vec![];
    }

    let n_features = points[0].len();
    let mut centroids = Vec::new();

    // Initialize centroids by randomly sampling from the actual data points
    let n_points = points.len();
    let mut used_indices = std::collections::HashSet::new();
    for _ in 0..k {
        if used_indices.len() >= n_points {
            break;
        }
        let mut idx = rng.gen_range(0..n_points);
        while used_indices.contains(&idx) {
            idx = rng.gen_range(0..n_points);
        }
        used_indices.insert(idx);
        centroids.push(points[idx].clone());
    }
    
    let actual_k = centroids.len();

    // Simple k-means iterations
    for _ in 0..20 {
        let mut new_centroids = vec![vec![0.0; n_features]; actual_k];
        let mut counts = vec![0usize; actual_k];

        for point in points {
            let closest_idx = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    euclidean_distance(point, a)
                        .partial_cmp(&euclidean_distance(point, b))
                        .unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            counts[closest_idx] += 1;
            for (i, &val) in point.iter().enumerate() {
                new_centroids[closest_idx][i] += val;
            }
        }

        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for feature in centroid.iter_mut() {
                    *feature /= counts[i] as f64;
                }
            } else {
                *centroid = centroids[i].clone();
            }
        }

        centroids = new_centroids;
    }

    centroids
}

fn kmeans_with_init(points: &[Vec<f64>], init_centroids: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if points.is_empty() {
        return vec![];
    }

    let n_features = points[0].len();
    let k = init_centroids.len();
    let mut centroids = init_centroids.to_vec();

    for _ in 0..20 {
        let mut new_centroids = vec![vec![0.0; n_features]; k];
        let mut counts = vec![0usize; k];

        for point in points {
            let closest_idx = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    euclidean_distance(point, a)
                        .partial_cmp(&euclidean_distance(point, b))
                        .unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            counts[closest_idx] += 1;
            for (i, &val) in point.iter().enumerate() {
                new_centroids[closest_idx][i] += val;
            }
        }

        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for feature in centroid.iter_mut() {
                    *feature /= counts[i] as f64;
                }
            } else {
                *centroid = centroids[i].clone();
            }
        }

        centroids = new_centroids;
    }

    centroids
}

#[pymodule]
fn clustream_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CluStream>()?;
    m.add_class::<CluStreamMicroCluster>()?;
    Ok(())
}
