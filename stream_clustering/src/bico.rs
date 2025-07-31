use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
struct ClusteringFeature {
    n: f64,
    linear_sum: Vec<f64>,
    squared_sum: f64,
    threshold: f64,
}

impl ClusteringFeature {
    fn new(point: &[f64], threshold: f64) -> Self {
        let squared_sum = point.iter().map(|x| x * x).sum();
        Self {
            n: 1.0,
            linear_sum: point.to_vec(),
            squared_sum,
            threshold,
        }
    }

    fn add(&mut self, weight: f64, point: &[f64], squared_norm: f64) {
        self.n += weight;
        for (i, &val) in point.iter().enumerate() {
            if i < self.linear_sum.len() {
                self.linear_sum[i] += weight * val;
            }
        }
        self.squared_sum += weight * squared_norm;
    }

    fn merge(&mut self, other: &ClusteringFeature) {
        self.n += other.n;
        for (i, &val) in other.linear_sum.iter().enumerate() {
            if i < self.linear_sum.len() {
                self.linear_sum[i] += val;
            }
        }
        self.squared_sum += other.squared_sum;
    }

    fn get_center(&self) -> Vec<f64> {
        if self.n > 0.0 {
            self.linear_sum.iter().map(|x| x / self.n).collect()
        } else {
            self.linear_sum.clone()
        }
    }

    fn calc_kmeans_costs(&self, center: &[f64], point: &[f64]) -> f64 {
        point.iter().zip(center.iter())
            .map(|(p, c)| (p - c).powi(2))
            .sum()
    }

    fn calc_kmeans_costs_cf(&self, center: &[f64]) -> f64 {
        let center_squared_norm: f64 = center.iter().map(|x| x * x).sum();
        let dot_product: f64 = self.linear_sum.iter().zip(center.iter())
            .map(|(ls, c)| ls * c).sum();
        
        self.squared_sum + self.n * center_squared_norm - 2.0 * dot_product
    }
}

#[derive(Clone)]
struct ClusteringTreeNode {
    center: Option<Vec<f64>>,
    clustering_feature: ClusteringFeature,
    children: Vec<ClusteringTreeNode>,
    threshold: f64,
}

impl ClusteringTreeNode {
    fn new(center: Option<Vec<f64>>, cf: ClusteringFeature) -> Self {
        Self {
            center,
            clustering_feature: cf,
            children: Vec::new(),
            threshold: 0.0,
        }
    }

    fn has_no_children(&self) -> bool {
        self.children.is_empty()
    }

    fn add_child(&mut self, child: ClusteringTreeNode) {
        self.children.push(child);
    }

    fn clear_children(&mut self) {
        self.children.clear();
    }

    fn get_center(&self) -> Vec<f64> {
        self.center.clone().unwrap_or_else(|| self.clustering_feature.get_center())
    }

    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
        for child in &mut self.children {
            child.set_threshold(threshold);
        }
    }

    fn nearest_child(&self, point: &[f64]) -> Option<usize> {
        if self.has_no_children() {
            return None;
        }

        let mut min_distance = f64::INFINITY;
        let mut nearest_idx = None;

        for (i, child) in self.children.iter().enumerate() {
            let center = child.get_center();
            let distance: f64 = point.iter().zip(center.iter())
                .map(|(p, c)| (p - c).powi(2)).sum();
            
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = Some(i);
            }
        }

        nearest_idx
    }

    fn add_to_clustering_centers(&self, centers: &mut Vec<Vec<f64>>) {
        if self.has_no_children() {
            let center = self.get_center();
            let mut weighted_center = vec![self.clustering_feature.n];
            weighted_center.extend(center);
            centers.push(weighted_center);
        } else {
            for child in &self.children {
                child.add_to_clustering_centers(centers);
            }
        }
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn kmeans_plus_plus_init(k: usize, points: &[Vec<f64>], rng: &mut StdRng) -> Vec<Vec<f64>> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut centroids = Vec::new();
    let weights: Vec<f64> = points.iter().map(|p| p[0]).collect();
    let point_coords: Vec<&[f64]> = points.iter().map(|p| &p[1..]).collect();

    // Choose first centroid randomly
    let total_weight: f64 = weights.iter().sum();
    let mut cumulative = 0.0;
    let rand_val = rng.gen::<f64>() * total_weight;
    
    for (i, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if cumulative >= rand_val {
            centroids.push(point_coords[i].to_vec());
            break;
        }
    }

    // Choose remaining centroids
    for _ in 1..k {
        let mut distances = Vec::new();
        for (i, point) in point_coords.iter().enumerate() {
            let min_dist = centroids.iter()
                .map(|c| euclidean_distance(point, c).powi(2))
                .fold(f64::INFINITY, f64::min);
            distances.push(weights[i] * min_dist);
        }

        let total_dist: f64 = distances.iter().sum();
        if total_dist == 0.0 {
            break;
        }

        let mut cumulative = 0.0;
        let rand_val = rng.gen::<f64>() * total_dist;
        
        for (i, &dist) in distances.iter().enumerate() {
            cumulative += dist;
            if cumulative >= rand_val {
                centroids.push(point_coords[i].to_vec());
                break;
            }
        }
    }

    centroids
}

#[pyclass]
pub struct BICO {
    n_clusters: usize,
    max_cluster_features: usize,
    n_projections: usize,
    
    buffer_phase: bool,
    buffer: Vec<Vec<f64>>,
    min_distance: f64,
    pairwise_different: usize,
    root: Option<ClusteringTreeNode>,
    root_count: usize,
    t_threshold: f64,
    n_dimensions: Option<usize>,
    
    rng: StdRng,
}

#[pymethods]
impl BICO {
    #[new]
    #[pyo3(signature = (n_clusters=5, max_cluster_features=1000, n_projections=10, seed=None))]
    fn new(
        n_clusters: usize,
        max_cluster_features: usize,
        n_projections: usize,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            n_clusters,
            max_cluster_features,
            n_projections,
            buffer_phase: true,
            buffer: Vec::new(),
            min_distance: f64::INFINITY,
            pairwise_different: 0,
            root: None,
            root_count: 0,
            t_threshold: 0.0,
            n_dimensions: None,
            rng,
        }
    }

    #[pyo3(signature = (x, _w=None))]
    fn learn_one(&mut self, x: Vec<f64>, _w: Option<f64>) {
        // Initialize dimensions on first point
        if self.n_dimensions.is_none() {
            self.n_dimensions = Some(x.len());
            let cf = ClusteringFeature::new(&[], 1.0);
            self.root = Some(ClusteringTreeNode::new(None, cf));
        }

        if let Some(n_dim) = self.n_dimensions {
            if x.len() != n_dim {
                return; // Skip points with wrong dimensions
            }
        }

        if self.buffer_phase {
            self.buffer_phase_update(x);
        } else {
            self.bico_update(&x);
        }
    }

    fn predict_one(&mut self, x: Vec<f64>) -> usize {
        let centers = self.get_centers();
        if centers.is_empty() {
            return 0;
        }

        let mut min_distance = f64::INFINITY;
        let mut nearest_cluster = 0;

        for (i, center) in centers.iter().enumerate() {
            let distance = euclidean_distance(&x, center);
            if distance < min_distance {
                min_distance = distance;
                nearest_cluster = i;
            }
        }

        nearest_cluster
    }

    fn get_centers(&self) -> Vec<Vec<f64>> {
        if self.buffer_phase || self.root.is_none() {
            return Vec::new();
        }

        let mut micro_clustering = Vec::new();
        
        if let Some(ref root) = self.root {
            root.add_to_clustering_centers(&mut micro_clustering);
        }

        if micro_clustering.is_empty() {
            return Vec::new();
        }

        // Apply k-means++ and Lloyd's algorithm
        let mut rng_clone = self.rng.clone();
        let centroids = kmeans_plus_plus_init(self.n_clusters, &micro_clustering, &mut rng_clone);
        self.kmeans_lloyd(centroids, &micro_clustering)
    }
}

impl BICO {
    fn buffer_phase_update(&mut self, point: Vec<f64>) {
        // Calculate pairwise distances to existing points
        for buffered_point in &self.buffer {
            let d = euclidean_distance(&point, buffered_point).powi(2);
            if d > 0.0 {
                self.pairwise_different += 1;
                if d < self.min_distance {
                    self.min_distance = d;
                }
            }
        }

        self.buffer.push(point);

        // Check if buffer is large enough
        if self.pairwise_different >= self.max_cluster_features + 1 {
            // Calculate starting threshold
            self.t_threshold = 16.0 * self.min_distance;
            let threshold = self.calc_r_squared(1);
            if let Some(ref mut root) = self.root {
                root.set_threshold(threshold);
            }
            self.buffer_phase = false;

            // Add all buffered points to the tree
            let buffer_copy = self.buffer.clone();
            self.buffer.clear();
            for buffered_point in buffer_copy {
                self.bico_update(&buffered_point);
            }
        }
    }

    fn bico_update(&mut self, point: &[f64]) {
        if self.root.is_none() {
            return;
        }

        let mut current_level = 1;
        let mut current_node_idx = vec![0]; // Path to current node

        loop {
            let nearest_child_idx = {
                let root = self.root.as_ref().unwrap();
                let current_node = Self::get_node_at_path(root, &current_node_idx);
                current_node.nearest_child(point)
            };

            if let Some(child_idx) = nearest_child_idx {
                let child_center = {
                    let root = self.root.as_ref().unwrap();
                    let mut child_path = current_node_idx.clone();
                    child_path.push(child_idx);
                    let child_node = Self::get_node_at_path(root, &child_path);
                    child_node.get_center()
                };

                let distance_squared = euclidean_distance(point, &child_center).powi(2);
                let r_squared = self.calc_r_squared(current_level);
                
                if distance_squared > r_squared {
                    // Create new node
                    let cf = ClusteringFeature::new(point, self.calc_r(current_level));
                    let new_node = ClusteringTreeNode::new(Some(point.to_vec()), cf);
                    
                    let root = self.root.as_mut().unwrap();
                    let current_node = Self::get_node_at_path_mut(root, &current_node_idx);
                    current_node.add_child(new_node);
                    self.root_count += 1;
                    break;
                } else {
                    // Check if point can be added without exceeding global threshold
                    let child_cf_cost = {
                        let root = self.root.as_ref().unwrap();
                        let mut child_path = current_node_idx.clone();
                        child_path.push(child_idx);
                        let child_node = Self::get_node_at_path(root, &child_path);
                        child_node.clustering_feature.calc_kmeans_costs(&child_center, point)
                    };

                    if child_cf_cost <= self.t_threshold {
                        // Add point to clustering feature
                        let squared_norm = point.iter().map(|x| x * x).sum();
                        let root = self.root.as_mut().unwrap();
                        let mut child_path = current_node_idx.clone();
                        child_path.push(child_idx);
                        let child_node = Self::get_node_at_path_mut(root, &child_path);
                        child_node.clustering_feature.add(1.0, point, squared_norm);
                        break;
                    } else {
                        // Navigate one level down
                        current_node_idx.push(child_idx);
                        current_level += 1;
                    }
                }
            } else {
                // No children, create new node
                let cf = ClusteringFeature::new(point, self.calc_r(current_level));
                let new_node = ClusteringTreeNode::new(Some(point.to_vec()), cf);
                
                let root = self.root.as_mut().unwrap();
                let current_node = Self::get_node_at_path_mut(root, &current_node_idx);
                current_node.add_child(new_node);
                self.root_count += 1;
                break;
            }
        }

        // Check if rebuild is needed
        if self.root_count > self.max_cluster_features {
            self.rebuild();
        }
    }

    fn rebuild(&mut self) {
        while self.root_count > self.max_cluster_features {
            // Double the global threshold
            self.t_threshold *= 2.0;
            let threshold = self.calc_r_squared(1);
            if let Some(ref mut root) = self.root {
                root.set_threshold(threshold);
            }

            // Collect all nodes and re-insert
            let mut queue = VecDeque::new();
            if let Some(ref mut root) = self.root {
                queue.extend(root.children.drain(..));
                self.root_count = 0;
            }

            while let Some(mut node) = queue.pop_front() {
                queue.extend(node.children.drain(..));
                self.bico_cf_update(node);
            }
        }
    }

    fn bico_cf_update(&mut self, node: ClusteringTreeNode) {
        // Similar to bico_update but for clustering features
        // Simplified implementation
        if let Some(ref mut root) = self.root {
            root.add_child(node);
            self.root_count += 1;
        }
    }

    fn calc_r(&self, level: i32) -> f64 {
        self.t_threshold / (4.0_f64.powi(level))
    }

    fn calc_r_squared(&self, level: i32) -> f64 {
        self.calc_r(level).powi(2)
    }

    fn get_node_at_path<'a>(root: &'a ClusteringTreeNode, path: &[usize]) -> &'a ClusteringTreeNode {
        let mut current = root;
        for &idx in path.iter().skip(1) {
            if idx < current.children.len() {
                current = &current.children[idx];
            }
        }
        current
    }

    fn get_node_at_path_mut<'a>(root: &'a mut ClusteringTreeNode, path: &[usize]) -> &'a mut ClusteringTreeNode {
        let mut current = root;
        for &idx in path.iter().skip(1) {
            if idx < current.children.len() {
                current = &mut current.children[idx];
            }
        }
        current
    }

    fn kmeans_lloyd(&self, mut centroids: Vec<Vec<f64>>, points: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if centroids.is_empty() || points.is_empty() {
            return Vec::new();
        }

        let weights: Vec<f64> = points.iter().map(|p| p[0]).collect();
        let point_coords: Vec<&[f64]> = points.iter().map(|p| &p[1..]).collect();

        for _ in 0..100 {
            // Assign points to nearest centroids
            let mut assignments = Vec::new();
            for point in &point_coords {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;
                
                for (k, centroid) in centroids.iter().enumerate() {
                    let distance = euclidean_distance(point, centroid).powi(2);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = k;
                    }
                }
                assignments.push(best_cluster);
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; centroids[0].len()]; centroids.len()];
            let mut cluster_weights = vec![0.0; centroids.len()];

            for (i, &assignment) in assignments.iter().enumerate() {
                let weight = weights[i];
                cluster_weights[assignment] += weight;
                for (j, &coord) in point_coords[i].iter().enumerate() {
                    new_centroids[assignment][j] += weight * coord;
                }
            }

            // Normalize centroids
            for (k, centroid) in new_centroids.iter_mut().enumerate() {
                if cluster_weights[k] > 0.0 {
                    for coord in centroid.iter_mut() {
                        *coord /= cluster_weights[k];
                    }
                } else {
                    *centroid = centroids[k].clone();
                }
            }

            // Check convergence
            let mut converged = true;
            for (old, new) in centroids.iter().zip(new_centroids.iter()) {
                if euclidean_distance(old, new) > 1e-6 {
                    converged = false;
                    break;
                }
            }

            centroids = new_centroids;
            if converged {
                break;
            }
        }

        centroids
    }
}

#[pymodule]
fn bico_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BICO>()?;
    Ok(())
}
