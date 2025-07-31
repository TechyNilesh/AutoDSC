use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::f64;

#[pyclass]
#[derive(Clone)]
pub struct DenStreamMicroCluster {
    #[pyo3(get)]
    pub creation_time: i64,
    #[pyo3(get)]
    pub last_edit_time: i64,
    #[pyo3(get)]
    pub decaying_factor: f64,
    #[pyo3(get)]
    pub n: i64,
    pub linear_sum: Vec<f64>,
    pub squared_sum: Vec<f64>,
    pub n_features: usize,
}

#[pymethods]
impl DenStreamMicroCluster {
    #[new]
    fn new(x: Vec<f64>, timestamp: i64, decaying_factor: f64) -> Self {
        let n_features = x.len();
        let squared_sum: Vec<f64> = x.iter().map(|&val| val * val).collect();
        
        Self {
            creation_time: timestamp,
            last_edit_time: timestamp,
            decaying_factor,
            n: 1,
            linear_sum: x,
            squared_sum,
            n_features,
        }
    }
    
    fn calc_weight(&self, timestamp: i64) -> f64 {
        let fading_factor = self.fading_function(timestamp - self.last_edit_time);
        self.weight(fading_factor)
    }
    
    fn calc_center(&self, timestamp: i64) -> Vec<f64> {
        let ff = self.fading_function(timestamp - self.last_edit_time);
        let weight = self.weight(ff);
        
        if weight == 0.0 {
            return vec![0.0; self.n_features];
        }
        
        self.linear_sum.iter().map(|&val| (ff * val) / weight).collect()
    }
    
    fn calc_radius(&self, timestamp: i64) -> f64 {
        let ff = self.fading_function(timestamp - self.last_edit_time);
        let weight = self.weight(ff);
        
        if weight == 0.0 {
            return 0.0;
        }
        
        let (norm_cf1, norm_cf2) = self.calc_norm_cf1_cf2(ff);
        let diff = (norm_cf2 / weight) - ((norm_cf1 / weight).powi(2));
        
        if diff > 0.0 {
            diff.sqrt()
        } else {
            0.0
        }
    }
    
    fn insert(&mut self, x: Vec<f64>, timestamp: i64) {
        self.n += 1;
        self.last_edit_time = timestamp;
        
        for i in 0..self.n_features.min(x.len()) {
            self.linear_sum[i] += x[i];
            self.squared_sum[i] += x[i] * x[i];
        }
    }
    
    fn merge(&mut self, other: &DenStreamMicroCluster) {
        self.n += other.n;
        
        for i in 0..self.n_features.min(other.n_features) {
            self.linear_sum[i] += other.linear_sum[i];
            self.squared_sum[i] += other.squared_sum[i];
        }
        
        if self.last_edit_time < other.creation_time {
            self.last_edit_time = other.creation_time;
        }
    }
}

impl DenStreamMicroCluster {
    fn calc_norm_cf1_cf2(&self, fading_factor: f64) -> (f64, f64) {
        let mut sum_of_squares_cf1 = 0.0;
        let mut sum_of_squares_cf2 = 0.0;
        
        for i in 0..self.n_features {
            let val_ls = self.linear_sum[i];
            let val_ss = self.squared_sum[i];
            sum_of_squares_cf1 += (fading_factor * val_ls).powi(2);
            sum_of_squares_cf2 += (fading_factor * val_ss).powi(2);
        }
        
        (sum_of_squares_cf1.sqrt(), sum_of_squares_cf2.sqrt())
    }
    
    fn weight(&self, fading_factor: f64) -> f64 {
        self.n as f64 * fading_factor
    }
    
    fn fading_function(&self, time: i64) -> f64 {
        2.0_f64.powf(-self.decaying_factor * time as f64)
    }
}

#[derive(Clone)]
struct BufferItem {
    x: Vec<f64>,
    timestamp: i64,
    covered: bool,
}

impl BufferItem {
    fn new(x: Vec<f64>, timestamp: i64) -> Self {
        Self {
            x,
            timestamp,
            covered: false,
        }
    }
}

#[pyclass]
pub struct DenStream {
    decaying_factor: f64,
    beta: f64,
    mu: f64,
    epsilon: f64,
    n_samples_init: usize,
    stream_speed: i64,
    
    timestamp: i64,
    initialized: bool,
    n_clusters: usize,
    clusters: HashMap<usize, DenStreamMicroCluster>,
    p_micro_clusters: HashMap<usize, DenStreamMicroCluster>,
    o_micro_clusters: HashMap<usize, DenStreamMicroCluster>,
    
    time_period: i64,
    init_buffer: VecDeque<BufferItem>,
    n_samples_seen: i64,
}

#[pymethods]
impl DenStream {
    #[new]
    #[pyo3(signature = (decaying_factor=0.01, beta=0.5, mu=3.0, epsilon=0.1, n_samples_init=100, stream_speed=1))]
    fn new(
        decaying_factor: f64,
        beta: f64,
        mu: f64,
        epsilon: f64,
        n_samples_init: usize,
        stream_speed: i64,
    ) -> PyResult<Self> {
        if !(0.0 < beta && beta <= 1.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("The value of `beta` (currently {}) must be within the range (0,1].", beta)
            ));
        }
        
        // Ensure mu * beta > 1.0 for valid time_period calculation
        if mu * beta <= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("The product mu * beta (currently {}) must be greater than 1.0.", mu * beta)
            ));
        }
        
        let time_period_calc = ((1.0 / decaying_factor) * ((mu * beta) / (mu * beta - 1.0)).ln()).ceil() as i64;
        let time_period = if time_period_calc <= 0 { 1 } else { time_period_calc };
        
        Ok(Self {
            decaying_factor,
            beta,
            mu,
            epsilon,
            n_samples_init,
            stream_speed,
            timestamp: -1,
            initialized: false,
            n_clusters: 0,
            clusters: HashMap::new(),
            p_micro_clusters: HashMap::new(),
            o_micro_clusters: HashMap::new(),
            time_period,
            init_buffer: VecDeque::new(),
            n_samples_seen: 0,
        })
    }
    
    #[pyo3(signature = (x, w=None))]
    fn learn_one(&mut self, x: Vec<f64>, w: Option<f64>) {
        self.n_samples_seen += 1;
        
        // Control the stream speed
        if self.n_samples_seen % self.stream_speed == 0 {
            self.timestamp += 1;
        }
        
        // Initialization
        if !self.initialized {
            self.init_buffer.push_back(BufferItem::new(x, self.timestamp));
            if self.init_buffer.len() == self.n_samples_init {
                self.initial_dbscan();
                self.initialized = true;
                self.init_buffer.clear();
            }
            return;
        }
        
        // Merge
        self.merge(x);
        
        // Periodic cluster removal
        if self.timestamp > 0 && self.timestamp % self.time_period == 0 {
            // Remove weak p-micro-clusters
            let mut p_clusters_to_remove = Vec::new();
            for (&i, p_micro_cluster) in &self.p_micro_clusters {
                if p_micro_cluster.calc_weight(self.timestamp) < self.mu * self.beta {
                    p_clusters_to_remove.push(i);
                }
            }
            for i in p_clusters_to_remove {
                self.p_micro_clusters.remove(&i);
            }
            
            // Remove weak o-micro-clusters
            let mut o_clusters_to_remove = Vec::new();
            for (&j, o_micro_cluster) in &self.o_micro_clusters {
                let xi = (2.0_f64.powf(-self.decaying_factor * (self.timestamp - o_micro_cluster.creation_time + self.time_period) as f64) - 1.0) /
                         (2.0_f64.powf(-self.decaying_factor * self.time_period as f64) - 1.0);
                if o_micro_cluster.calc_weight(self.timestamp) < xi {
                    o_clusters_to_remove.push(j);
                }
            }
            for j in o_clusters_to_remove {
                self.o_micro_clusters.remove(&j);
            }
        }
    }
    
    #[pyo3(signature = (x, w=None))]
    fn predict_one(&mut self, x: Vec<f64>, w: Option<f64>) -> usize {
        if !self.initialized {
            return 0;
        }
        
        // Apply DBSCAN on p-micro-clusters
        let mut c = 0;
        let mut labels: HashMap<usize, Option<usize>> = self.p_micro_clusters.keys()
            .map(|&k| (k, None))
            .collect();
        
        for &pmc_key in self.p_micro_clusters.keys() {
            if labels[&pmc_key].is_some() {
                continue;
            }
            
            labels.insert(pmc_key, Some(c));
            let mut seed_queue = self.query_neighbor(pmc_key);
            
            while let Some(neighbor_key) = seed_queue.pop_front() {
                if labels[&neighbor_key].is_some() {
                    continue;
                }
                
                labels.insert(neighbor_key, Some(c));
                let neighbor_neighbors = self.query_neighbor(neighbor_key);
                
                for nn_key in neighbor_neighbors {
                    if labels[&nn_key].is_none() {
                        seed_queue.push_back(nn_key);
                    }
                }
            }
            
            c += 1;
        }
        
        // Generate clusters from labels
        self.generate_clusters_from_labels(&labels);
        
        self.get_closest_cluster_key(&x, &self.clusters)
    }
    
    fn get_n_clusters(&self) -> usize {
        self.n_clusters
    }
    
    fn get_centers(&self) -> Vec<Vec<f64>> {
        self.clusters.values()
            .map(|cluster| cluster.calc_center(self.timestamp))
            .collect()
    }
    
    fn get_p_micro_clusters(&self) -> Vec<DenStreamMicroCluster> {
        self.p_micro_clusters.values().cloned().collect()
    }
    
    fn get_o_micro_clusters(&self) -> Vec<DenStreamMicroCluster> {
        self.o_micro_clusters.values().cloned().collect()
    }
}

impl DenStream {
    fn distance(&self, point_a: &[f64], point_b: &[f64]) -> f64 {
        euclidean_distance(point_a, point_b)
    }
    
    fn get_closest_cluster_key(&self, point: &[f64], clusters: &HashMap<usize, DenStreamMicroCluster>) -> usize {
        let mut min_distance = f64::INFINITY;
        let mut key = 0;
        
        for (&k, cluster) in clusters {
            let center = cluster.calc_center(self.timestamp);
            let distance = self.distance(&center, point);
            if distance < min_distance {
                min_distance = distance;
                key = k;
            }
        }
        
        key
    }
    
    fn merge(&mut self, point: Vec<f64>) {
        let mut merged_status = false;
        
        // Try to merge with p-micro-clusters
        if !self.p_micro_clusters.is_empty() {
            let closest_pmc_key = self.get_closest_cluster_key(&point, &self.p_micro_clusters);
            
            if let Some(pmc) = self.p_micro_clusters.get(&closest_pmc_key) {
                let mut updated_pmc = pmc.clone();
                updated_pmc.insert(point.clone(), self.timestamp);
                
                if updated_pmc.calc_radius(self.timestamp) <= self.epsilon {
                    self.p_micro_clusters.insert(closest_pmc_key, updated_pmc);
                    merged_status = true;
                }
            }
        }
        
        // Try to merge with o-micro-clusters if not merged with p-micro-clusters
        if !merged_status && !self.o_micro_clusters.is_empty() {
            let closest_omc_key = self.get_closest_cluster_key(&point, &self.o_micro_clusters);
            
            if let Some(omc) = self.o_micro_clusters.get(&closest_omc_key) {
                let mut updated_omc = omc.clone();
                updated_omc.insert(point.clone(), self.timestamp);
                
                if updated_omc.calc_radius(self.timestamp) <= self.epsilon {
                    if updated_omc.calc_weight(self.timestamp) > self.mu * self.beta {
                        // It has grown into a p-micro-cluster
                        self.o_micro_clusters.remove(&closest_omc_key);
                        let new_key = self.p_micro_clusters.len();
                        self.p_micro_clusters.insert(new_key, updated_omc);
                    } else {
                        self.o_micro_clusters.insert(closest_omc_key, updated_omc);
                    }
                } else {
                    // Create a new o-micro-cluster
                    let mc_from_p = DenStreamMicroCluster::new(point.clone(), self.timestamp, self.decaying_factor);
                    let new_key = self.o_micro_clusters.len();
                    self.o_micro_clusters.insert(new_key, mc_from_p);
                }
                merged_status = true;
            }
        }
        
        // Create new o-micro-cluster if not merged and o-micro-cluster set is empty
        if !merged_status {
            let mc_from_p = DenStreamMicroCluster::new(point, self.timestamp, self.decaying_factor);
            self.o_micro_clusters.insert(0, mc_from_p);
        }
    }
    
    fn is_directly_density_reachable(&self, c_p_key: usize, c_q_key: usize) -> bool {
        if let (Some(c_p), Some(c_q)) = (self.p_micro_clusters.get(&c_p_key), self.p_micro_clusters.get(&c_q_key)) {
            if c_p.calc_weight(self.timestamp) > self.mu && c_q.calc_weight(self.timestamp) > self.mu {
                let c_p_center = c_p.calc_center(self.timestamp);
                let c_q_center = c_q.calc_center(self.timestamp);
                let distance = self.distance(&c_p_center, &c_q_center);
                
                return distance < 2.0 * self.epsilon && 
                       distance <= c_p.calc_radius(self.timestamp) + c_q.calc_radius(self.timestamp);
            }
        }
        false
    }
    
    fn query_neighbor(&self, cluster_key: usize) -> VecDeque<usize> {
        let mut neighbors = VecDeque::new();
        
        for &pmc_key in self.p_micro_clusters.keys() {
            if cluster_key != pmc_key && self.is_directly_density_reachable(cluster_key, pmc_key) {
                neighbors.push_back(pmc_key);
            }
        }
        
        neighbors
    }
    
    fn generate_clusters_from_labels(&mut self, cluster_labels: &HashMap<usize, Option<usize>>) {
        let mut clusters = HashMap::new();
        let mut mcs_per_label: HashMap<usize, Vec<&DenStreamMicroCluster>> = HashMap::new();
        
        // Group micro-clusters by label
        for (&mc_key, &label_opt) in cluster_labels {
            if let (Some(label), Some(mc)) = (label_opt, self.p_micro_clusters.get(&mc_key)) {
                mcs_per_label.entry(label).or_insert_with(Vec::new).push(mc);
            }
        }
        
        // Generate clusters by merging micro-clusters with the same label
        for (label, micro_clusters) in mcs_per_label {
            if !micro_clusters.is_empty() {
                let mut cluster = micro_clusters[0].clone();
                for mc in micro_clusters.iter().skip(1) {
                    cluster.merge(mc);
                }
                clusters.insert(label, cluster);
            }
        }
        
        self.n_clusters = clusters.len();
        self.clusters = clusters;
    }
    
    fn expand_cluster(&self, mc: &mut DenStreamMicroCluster, neighborhood: &[usize]) {
        for &idx in neighborhood {
            if let Some(item) = self.init_buffer.get(idx) {
                if !item.covered {
                    mc.insert(item.x.clone(), self.timestamp);
                }
            }
        }
    }
    
    fn get_neighborhood_ids(&self, item: &BufferItem) -> Vec<usize> {
        let mut neighborhood_ids = Vec::new();
        
        for (idx, other) in self.init_buffer.iter().enumerate() {
            if !other.covered && self.distance(&item.x, &other.x) < self.epsilon {
                neighborhood_ids.push(idx);
            }
        }
        
        neighborhood_ids
    }
    
    fn initial_dbscan(&mut self) {
        let mut buffer_items: Vec<BufferItem> = self.init_buffer.iter().cloned().collect();
        
        for (idx, item) in buffer_items.iter_mut().enumerate() {
            if !item.covered {
                item.covered = true;
                let neighborhood = self.get_neighborhood_ids(item);
                
                if neighborhood.len() as f64 > self.mu {
                    let mut mc = DenStreamMicroCluster::new(item.x.clone(), self.timestamp, self.decaying_factor);
                    self.expand_cluster(&mut mc, &neighborhood);
                    let new_key = self.p_micro_clusters.len();
                    self.p_micro_clusters.insert(new_key, mc);
                } else {
                    item.covered = false;
                }
            }
        }
        
        // Update the buffer with the modified items
        self.init_buffer = buffer_items.into();
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
fn denstream_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DenStream>()?;
    m.add_class::<DenStreamMicroCluster>()?;
    Ok(())
}
