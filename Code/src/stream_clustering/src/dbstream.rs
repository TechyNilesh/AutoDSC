use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::f64;

#[pyclass]
#[derive(Clone)]
pub struct DBStreamMicroCluster {
    #[pyo3(get)]
    pub center: Vec<f64>,
    #[pyo3(get)]
    pub last_update: i64,
    #[pyo3(get)]
    pub weight: f64,
}

#[pymethods]
impl DBStreamMicroCluster {
    #[new]
    fn new(x: Vec<f64>, last_update: i64, weight: f64) -> Self {
        Self {
            center: x,
            last_update,
            weight,
        }
    }
    
    fn merge(&mut self, other: &DBStreamMicroCluster) {
        let total_weight = self.weight + other.weight;
        
        for i in 0..self.center.len().min(other.center.len()) {
            self.center[i] = (self.center[i] * self.weight + other.center[i] * other.weight) / total_weight;
        }
        
        self.weight = total_weight;
    }
}

#[pyclass]
pub struct DBStream {
    clustering_threshold: f64,
    fading_factor: f64,
    cleanup_interval: i64,
    intersection_factor: f64,
    minimum_weight: f64,
    
    time_stamp: i64,
    n_clusters: usize,
    clusters: HashMap<usize, DBStreamMicroCluster>,
    centers: Vec<Vec<f64>>,
    micro_clusters: HashMap<usize, DBStreamMicroCluster>,
    
    // Shared density matrices
    s: HashMap<usize, HashMap<usize, f64>>,
    s_t: HashMap<usize, HashMap<usize, i64>>,
    
    clustering_is_up_to_date: bool,
}

#[pymethods]
impl DBStream {
    #[new]
    #[pyo3(signature = (clustering_threshold=1.0, fading_factor=0.01, cleanup_interval=1000.0, intersection_factor=0.3, minimum_weight=1.0))]
    fn new(
        clustering_threshold: f64,
        fading_factor: f64,
        cleanup_interval: f64,
        intersection_factor: f64,
        minimum_weight: f64,
    ) -> Self {
        Self {
            clustering_threshold,
            fading_factor,
            cleanup_interval: cleanup_interval as i64,
            intersection_factor,
            minimum_weight,
            time_stamp: 0,
            n_clusters: 0,
            clusters: HashMap::new(),
            centers: Vec::new(),
            micro_clusters: HashMap::new(),
            s: HashMap::new(),
            s_t: HashMap::new(),
            clustering_is_up_to_date: false,
        }
    }
    
    #[pyo3(signature = (x, w=None))]
    fn learn_one(&mut self, x: Vec<f64>, w: Option<f64>) {
        self.update(x);
        
        if self.time_stamp % self.cleanup_interval == 0 {
            self.cleanup();
        }
        
        self.clustering_is_up_to_date = false;
    }
    
    #[pyo3(signature = (x, w=None))]
    fn predict_one(&mut self, x: Vec<f64>, w: Option<f64>) -> usize {
        self.recluster();
        
        let mut min_distance = f64::INFINITY;
        let mut closest_cluster_index = 0;
        
        for (i, center) in self.centers.iter().enumerate() {
            let distance = euclidean_distance(&x, center);
            if distance < min_distance {
                min_distance = distance;
                closest_cluster_index = i;
            }
        }
        
        closest_cluster_index
    }
    
    fn get_n_clusters(&mut self) -> usize {
        self.recluster();
        self.n_clusters
    }
    
    fn get_centers(&mut self) -> Vec<Vec<f64>> {
        self.recluster();
        self.centers.clone()
    }
    
    fn get_micro_clusters(&self) -> Vec<DBStreamMicroCluster> {
        self.micro_clusters.values().cloned().collect()
    }
}

impl DBStream {
    fn distance(&self, point_a: &[f64], point_b: &[f64]) -> f64 {
        euclidean_distance(point_a, point_b)
    }
    
    fn find_fixed_radius_nn(&self, x: &[f64]) -> HashMap<usize, DBStreamMicroCluster> {
        let mut fixed_radius_nn = HashMap::new();
        
        for (&i, micro_cluster) in &self.micro_clusters {
            if self.distance(&micro_cluster.center, x) < self.clustering_threshold {
                fixed_radius_nn.insert(i, micro_cluster.clone());
            }
        }
        
        fixed_radius_nn
    }
    
    fn gaussian_neighborhood(&self, point_a: &[f64], point_b: &[f64]) -> f64 {
        let distance = self.distance(point_a, point_b);
        let sigma = self.clustering_threshold / 3.0;
        (-distance * distance / (2.0 * sigma * sigma)).exp()
    }
    
    fn update(&mut self, x: Vec<f64>) {
        let neighbor_clusters = self.find_fixed_radius_nn(&x);
        
        if neighbor_clusters.is_empty() {
            // Create new micro cluster
            let new_id = if self.micro_clusters.is_empty() {
                0
            } else {
                self.micro_clusters.keys().max().unwrap() + 1
            };
            
            self.micro_clusters.insert(
                new_id,
                DBStreamMicroCluster::new(x, self.time_stamp, 1.0)
            );
        } else {
            // Update existing micro clusters
            let mut current_centers = HashMap::new();
            let neighbor_keys: Vec<usize> = neighbor_clusters.keys().cloned().collect();
             for &i in &neighbor_keys {
                if let Some(micro_cluster) = self.micro_clusters.get_mut(&i) {
                    current_centers.insert(i, micro_cluster.center.clone());
                    
                    // Apply fading and update weight
                    let fading_factor = 2.0_f64.powf(-self.fading_factor * (self.time_stamp - micro_cluster.last_update) as f64);
                    micro_cluster.weight = micro_cluster.weight * fading_factor + 1.0;
                    
                    micro_cluster.last_update = self.time_stamp;
                }
            }
            
            // Update centers using Gaussian neighborhood function (separate loop to avoid borrow checker issues)
            for &i in &neighbor_keys {
                let center_copy = if let Some(micro_cluster) = self.micro_clusters.get(&i) {
                    micro_cluster.center.clone()
                } else {
                    continue;
                };
                
                let gaussian_weight = self.gaussian_neighborhood(&x, &center_copy);
                
                if let Some(micro_cluster) = self.micro_clusters.get_mut(&i) {
                    for j in 0..micro_cluster.center.len().min(x.len()) {
                        micro_cluster.center[j] += gaussian_weight * (x[j] - micro_cluster.center[j]);
                    }
                }
            }
            
            // Update shared density
            for &i in &neighbor_keys {
                for &j in &neighbor_keys {
                    if j > i {
                        let shared_density_update = if let Some(s_ij) = self.s.get(&i).and_then(|s_i| s_i.get(&j)) {
                            let s_t_ij = self.s_t.get(&i).and_then(|s_t_i| s_t_i.get(&j)).unwrap_or(&0);
                            let fading = 2.0_f64.powf(-self.fading_factor * (self.time_stamp - s_t_ij) as f64);
                            s_ij * fading + 1.0
                        } else {
                            1.0
                        };
                        
                        self.s.entry(i).or_insert_with(HashMap::new).insert(j, shared_density_update);
                        self.s_t.entry(i).or_insert_with(HashMap::new).insert(j, self.time_stamp);
                    }
                }
            }
            
            // Prevent collapsing clusters
            for &i in &neighbor_keys {
                for &j in &neighbor_keys {
                    if j > i {
                        if let (Some(mc_i), Some(mc_j)) = (self.micro_clusters.get(&i), self.micro_clusters.get(&j)) {
                            if self.distance(&mc_i.center, &mc_j.center) < self.clustering_threshold {
                                // Revert centers to previous positions
                                if let (Some(prev_center_i), Some(prev_center_j)) = (current_centers.get(&i), current_centers.get(&j)) {
                                    if let Some(mc_i) = self.micro_clusters.get_mut(&i) {
                                        mc_i.center = prev_center_i.clone();
                                    }
                                    if let Some(mc_j) = self.micro_clusters.get_mut(&j) {
                                        mc_j.center = prev_center_j.clone();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        self.time_stamp += 1;
    }
    
    fn cleanup(&mut self) {
        let weight_weak = 2.0_f64.powf(-self.fading_factor * self.cleanup_interval as f64);
        
        // Remove weak micro clusters
        let mut clusters_to_remove = Vec::new();
        for (&i, micro_cluster) in &self.micro_clusters {
            let fading = 2.0_f64.powf(-self.fading_factor * (self.time_stamp - micro_cluster.last_update) as f64);
            if micro_cluster.weight * fading < weight_weak {
                clusters_to_remove.push(i);
            }
        }
        
        for i in clusters_to_remove {
            self.micro_clusters.remove(&i);
            self.s.remove(&i);
            self.s_t.remove(&i);
            
            // Remove references to this cluster from other clusters' shared density maps
            for s_map in self.s.values_mut() {
                s_map.remove(&i);
            }
            for s_t_map in self.s_t.values_mut() {
                s_t_map.remove(&i);
            }
        }
        
        // Clean up weak shared density entries
        let mut entries_to_remove = Vec::new();
        for (&i, s_i) in &self.s {
            for (&j, &s_ij) in s_i {
                if let Some(s_t_ij) = self.s_t.get(&i).and_then(|s_t_i| s_t_i.get(&j)) {
                    let fading = 2.0_f64.powf(-self.fading_factor * (self.time_stamp - s_t_ij) as f64);
                    if s_ij * fading < self.intersection_factor * weight_weak {
                        entries_to_remove.push((i, j));
                    }
                }
            }
        }
        
        for (i, j) in entries_to_remove {
            if let Some(s_i) = self.s.get_mut(&i) {
                s_i.insert(j, 0.0);
            }
            if let Some(s_t_i) = self.s_t.get_mut(&i) {
                s_t_i.insert(j, 0);
            }
        }
    }
    
    fn generate_weighted_adjacency_matrix(&self) -> HashMap<usize, HashMap<usize, f64>> {
        let mut weighted_adjacency_matrix = HashMap::new();
        
        for (&i, s_i) in &self.s {
            for (&j, &s_ij) in s_i {
                if let (Some(mc_i), Some(mc_j)) = (self.micro_clusters.get(&i), self.micro_clusters.get(&j)) {
                    if mc_i.weight <= self.minimum_weight || mc_j.weight <= self.minimum_weight {
                        continue;
                    }
                    
                    let value = s_ij / ((mc_i.weight + mc_j.weight) / 2.0);
                    if value > self.intersection_factor {
                        weighted_adjacency_matrix.entry(i).or_insert_with(HashMap::new).insert(j, value);
                    }
                }
            }
        }
        
        weighted_adjacency_matrix
    }
    
    fn generate_labels(&self, weighted_adjacency_list: &HashMap<usize, HashMap<usize, f64>>) -> HashMap<usize, usize> {
        // Only strong micro-clusters (weight > minimum_weight) form clusters;
        // weak ones are treated as noise and excluded from macro-clusters.
        let mut strong: Vec<usize> = self
            .micro_clusters
            .iter()
            .filter(|(_, mc)| mc.weight >= self.minimum_weight)
            .map(|(&k, _)| k)
            .collect();
        // Deterministic label assignment: Rust HashMap iteration is randomized,
        // so sort keys (mimics Python dict insertion order) to keep labels stable
        // across consecutive recluster() calls.
        strong.sort_unstable();

        let mut labels: HashMap<usize, Option<usize>> = strong.iter().map(|&k| (k, None)).collect();
        let mut count = 0;

        for &index in &strong {
            if labels[&index].is_some() {
                continue;
            }

            labels.insert(index, Some(count));

            if let Some(neighbors) = weighted_adjacency_list.get(&index) {
                let mut seed_set: VecDeque<usize> = neighbors.keys().cloned().collect();

                while let Some(current) = seed_set.pop_front() {
                    // skip weak/noise micro-clusters not in the strong label set
                    if !labels.contains_key(&current) || labels[&current].is_some() {
                        continue;
                    }

                    labels.insert(current, Some(count));

                    if let Some(neighbor_neighbors) = weighted_adjacency_list.get(&current) {
                        for &neighbor_neighbor in neighbor_neighbors.keys() {
                            if labels.get(&neighbor_neighbor).map_or(false, |l| l.is_none()) {
                                seed_set.push_back(neighbor_neighbor);
                            }
                        }
                    }
                }
            }

            count += 1;
        }

        labels.into_iter().map(|(k, v)| (k, v.unwrap_or(0))).collect()
    }
    
    fn generate_clusters_from_labels(&self, cluster_labels: &HashMap<usize, usize>) -> (usize, HashMap<usize, DBStreamMicroCluster>) {
        let mut clusters = HashMap::new();
        
        if cluster_labels.is_empty() {
            return (0, clusters);
        }
        
        let max_label = *cluster_labels.values().max().unwrap();
        
        for i in 0..=max_label {
            let mut mcs_with_label_i: Vec<&DBStreamMicroCluster> = Vec::new();
            
            for (&index, &label) in cluster_labels {
                if label == i {
                    if let Some(mc) = self.micro_clusters.get(&index) {
                        mcs_with_label_i.push(mc);
                    }
                }
            }
            
            if !mcs_with_label_i.is_empty() {
                let mut macro_cluster = mcs_with_label_i[0].clone();
                
                for mc in mcs_with_label_i.iter().skip(1) {
                    macro_cluster.merge(mc);
                }
                
                clusters.insert(i, macro_cluster);
            }
        }
        
        let n_clusters = clusters.len();
        (n_clusters, clusters)
    }
    
    fn recluster(&mut self) {
        if self.clustering_is_up_to_date {
            return;
        }
        
        let weighted_adjacency_list = self.generate_weighted_adjacency_matrix();
        let labels = self.generate_labels(&weighted_adjacency_list);
        
        if !labels.is_empty() {
            let (n_clusters, clusters) = self.generate_clusters_from_labels(&labels);
            self.n_clusters = n_clusters;
            self.clusters = clusters;
            // Build centers in sorted label order so predict_one's index maps to a
            // stable cluster label (HashMap value order is otherwise randomized).
            let mut keys: Vec<usize> = self.clusters.keys().cloned().collect();
            keys.sort_unstable();
            self.centers = keys.iter().map(|k| self.clusters[k].center.clone()).collect();
        }
        
        self.clustering_is_up_to_date = true;
    }
}

#[pymodule]
fn dbstream_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DBStream>()?;
    m.add_class::<DBStreamMicroCluster>()?;
    Ok(())
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
