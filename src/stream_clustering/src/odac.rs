use pyo3::prelude::*;
use std::collections::HashMap;
use std::f64;

#[pyclass]
#[derive(Clone)]
pub struct ODACChildren {
    pub first: Box<ODACCluster>,
    pub second: Box<ODACCluster>,
}

#[pymethods]
impl ODACChildren {
    #[new]
    fn new(first: ODACCluster, second: ODACCluster) -> Self {
        Self {
            first: Box::new(first),
            second: Box::new(second),
        }
    }
    
    fn reset_parent(&mut self) {
        self.first.parent = None;
        self.second.parent = None;
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PearsonCorr {
    n: i64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

#[pymethods]
impl PearsonCorr {
    #[new]
    fn new() -> Self {
        Self {
            n: 0,
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xx: 0.0,
            sum_yy: 0.0,
            sum_xy: 0.0,
        }
    }
    
    fn update(&mut self, x: f64, y: f64) {
        self.n += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }
    
    fn get(&self) -> f64 {
        if self.n <= 1 {
            return 0.0;
        }
        
        let n = self.n as f64;
        let numerator = n * self.sum_xy - self.sum_x * self.sum_y;
        let denominator = ((n * self.sum_xx - self.sum_x * self.sum_x) * 
                          (n * self.sum_yy - self.sum_y * self.sum_y)).sqrt();
        
        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Var {
    n: i64,
    sum_x: f64,
    sum_xx: f64,
}

#[pymethods]
impl Var {
    #[new]
    fn new() -> Self {
        Self {
            n: 0,
            sum_x: 0.0,
            sum_xx: 0.0,
        }
    }
    
    fn update(&mut self, x: f64) {
        self.n += 1;
        self.sum_x += x;
        self.sum_xx += x * x;
    }
    
    fn get(&self) -> f64 {
        if self.n <= 1 {
            return 0.0;
        }
        
        let n = self.n as f64;
        let mean = self.sum_x / n;
        (self.sum_xx / n) - (mean * mean)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ODACCluster {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub active: bool,
    #[pyo3(get)]
    pub timeseries_names: Vec<String>,
    #[pyo3(get)]
    pub n: i64,
    #[pyo3(get)]
    pub d1: Option<f64>,
    #[pyo3(get)]
    pub d2: Option<f64>,
    #[pyo3(get)]
    pub e: f64,
    
    pub parent: Option<Box<ODACCluster>>,
    pub children: Option<ODACChildren>,
    pub d0: Option<f64>,
    pub avg: Option<f64>,
    pub pivot_0: Option<(String, String)>,
    pub pivot_1: Option<(String, String)>,
    pub pivot_2: Option<(String, String)>,
    
    // Statistics - either correlations for multiple time series or variance for single
    pub correlations: HashMap<(String, String), PearsonCorr>,
    pub variance: Option<Var>,
}

#[pymethods]
impl ODACCluster {
    #[new]
    fn new(name: String) -> Self {
        Self {
            name,
            active: true,
            timeseries_names: Vec::new(),
            n: 0,
            d1: None,
            d2: None,
            e: 0.0,
            parent: None,
            children: None,
            d0: None,
            avg: None,
            pivot_0: None,
            pivot_1: None,
            pivot_2: None,
            correlations: HashMap::new(),
            variance: None,
        }
    }
    
    fn initialize_timeseries(&mut self, ts_names: Vec<String>) {
        let mut sorted_names = ts_names;
        sorted_names.sort();
        self.timeseries_names = sorted_names;
        self.init_stats();
    }
    
    fn update_statistics(&mut self, x: HashMap<String, f64>) {
        if self.timeseries_names.len() > 1 {
            // Update correlations for each pair
            for i in 0..self.timeseries_names.len() {
                for j in (i + 1)..self.timeseries_names.len() {
                    let k1 = &self.timeseries_names[i];
                    let k2 = &self.timeseries_names[j];
                    
                    if let (Some(&v1), Some(&v2)) = (x.get(k1), x.get(k2)) {
                        let key = (k1.clone(), k2.clone());
                        self.correlations.entry(key).or_insert_with(PearsonCorr::new).update(v1, v2);
                    }
                }
            }
        } else if let Some(ts_name) = self.timeseries_names.first() {
            if let Some(&value) = x.get(ts_name) {
                if let Some(ref mut var) = self.variance {
                    var.update(value);
                }
            }
        }
        
        self.n += 1;
    }
    
    fn calculate_coefficients(&mut self, confidence_level: f64) {
        if self.timeseries_names.len() > 1 {
            let rnomc_dict = self.calculate_rnomc_dict();
            
            // Calculate average distance
            if !rnomc_dict.is_empty() {
                self.avg = Some(rnomc_dict.values().sum::<f64>() / self.n as f64);
            }
            
            // Find minimum distance (d0) and its pivot
            if let Some((pivot, &dist)) = rnomc_dict.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                self.pivot_0 = Some(pivot.clone());
                self.d0 = Some(dist);
            }
            
            // Find maximum distance (d1) and its pivot
            if let Some((pivot, &dist)) = rnomc_dict.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                self.pivot_1 = Some(pivot.clone());
                self.d1 = Some(dist);
            }
            
            // Find second maximum distance (d2) and its pivot
            if let Some(pivot_1) = &self.pivot_1 {
                if let Some((pivot, &dist)) = rnomc_dict.iter()
                    .filter(|(k, _)| *k != pivot_1)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                    self.pivot_2 = Some(pivot.clone());
                    self.d2 = Some(dist);
                }
            }
        } else if let Some(ref var) = self.variance {
            self.d1 = Some(var.get());
        }
        
        // Calculate Hoeffding bound
        self.e = (((1.0 / confidence_level).ln()) / (2.0 * self.n as f64)).sqrt();
    }
    
    fn test_split(&mut self, tau: f64) -> bool {
        if let (Some(d1), Some(d2)) = (self.d1, self.d2) {
            if (d1 - d2) > self.e || tau > self.e {
                if let (Some(d0), Some(avg)) = (self.d0, self.avg) {
                    if (d1 - d0) * (d1 + d0 - 2.0 * avg).abs() > self.e {
                        if let Some(pivot_1) = &self.pivot_1 {
                            self.split_this_cluster(pivot_1.0.clone(), pivot_1.1.clone());
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
    
    fn test_aggregate(&mut self) -> bool {
        if let (Some(ref parent), Some(d1)) = (&self.parent, self.d1) {
            if let Some(parent_d1) = parent.d1 {
                if d1 - parent_d1 > parent.e.max(self.e) {
                    // Note: In a real implementation, we'd need to access the parent mutably
                    // This is a simplified version
                    return true;
                }
            }
        }
        false
    }
    
    fn design_structure(&self, decimal_places: usize) -> String {
        let mut result = String::new();
        
        // Format d1
        let d1_str = if let Some(d1) = self.d1 {
            format!("{:.prec$}", d1, prec = decimal_places)
        } else {
            "<Not calculated>".to_string()
        };
        
        // Format d2
        let d2_str = if let Some(d2) = self.d2 {
            format!(" d2={:.prec$}", d2, prec = decimal_places)
        } else {
            "".to_string()
        };
        
        result.push_str(&format!("{} d1={}{}", self.name, d1_str, d2_str));
        
        if self.active {
            result.push_str(&format!(" {:?}", self.timeseries_names));
        } else {
            result.push_str(" [NOT ACTIVE]");
        }
        
        result
    }
}

impl ODACCluster {
    fn init_stats(&mut self) {
        if self.timeseries_names.len() > 1 {
            self.correlations.clear();
            for i in 0..self.timeseries_names.len() {
                for j in (i + 1)..self.timeseries_names.len() {
                    let key = (self.timeseries_names[i].clone(), self.timeseries_names[j].clone());
                    self.correlations.insert(key, PearsonCorr::new());
                }
            }
        } else {
            self.variance = Some(Var::new());
        }
    }
    
    fn calculate_rnomc_dict(&self) -> HashMap<(String, String), f64> {
        let mut rnomc_dict = HashMap::new();
        
        for ((k1, k2), corr) in &self.correlations {
            let corr_value = corr.get();
            let rnomc_value = ((1.0 - corr_value).abs() / 2.0).sqrt();
            rnomc_dict.insert((k1.clone(), k2.clone()), rnomc_value);
        }
        
        rnomc_dict
    }
    
    fn get_closest_cluster(&self, pivot_1: &str, pivot_2: &str, current: &str, rnomc_dict: &HashMap<(String, String), f64>) -> i32 {
        let key1 = if pivot_1 < current {
            (pivot_1.to_string(), current.to_string())
        } else {
            (current.to_string(), pivot_1.to_string())
        };
        
        let key2 = if pivot_2 < current {
            (pivot_2.to_string(), current.to_string())
        } else {
            (current.to_string(), pivot_2.to_string())
        };
        
        let dist_1 = rnomc_dict.get(&key1).unwrap_or(&0.0);
        let dist_2 = rnomc_dict.get(&key2).unwrap_or(&0.0);
        
        if dist_1 >= dist_2 { 2 } else { 1 }
    }
    
    fn split_this_cluster(&mut self, pivot_1: String, pivot_2: String) {
        let rnomc_dict = self.calculate_rnomc_dict();
        let mut pivot_1_list = vec![pivot_1.clone()];
        let mut pivot_2_list = vec![pivot_2.clone()];
        
        for ts_name in &self.timeseries_names {
            if ts_name != &pivot_1 && ts_name != &pivot_2 {
                let cluster = self.get_closest_cluster(&pivot_1, &pivot_2, ts_name, &rnomc_dict);
                if cluster == 1 {
                    pivot_1_list.push(ts_name.clone());
                } else {
                    pivot_2_list.push(ts_name.clone());
                }
            }
        }
        
        let new_name = if self.name == "ROOT" {
            "1".to_string()
        } else {
            let parts: Vec<&str> = self.name.split('_').collect();
            if let Some(last) = parts.last() {
                if let Ok(num) = last.parse::<i32>() {
                    (num + 1).to_string()
                } else {
                    "1".to_string()
                }
            } else {
                "1".to_string()
            }
        };
        
        let mut cluster_child_1 = ODACCluster::new(format!("CH1_LVL_{}", new_name));
        cluster_child_1.initialize_timeseries(pivot_1_list);
        
        let mut cluster_child_2 = ODACCluster::new(format!("CH2_LVL_{}", new_name));
        cluster_child_2.initialize_timeseries(pivot_2_list);
        
        self.children = Some(ODACChildren::new(cluster_child_1, cluster_child_2));
        self.active = false;
        
        // Reset attributes
        self.avg = None;
        self.d0 = None;
        self.pivot_0 = None;
        self.pivot_1 = None;
        self.pivot_2 = None;
        self.correlations.clear();
        self.variance = None;
    }
    
    fn aggregate_this_cluster(&mut self) {
        self.init_stats();
        self.active = true;
        
        if let Some(ref mut children) = self.children {
            children.reset_parent();
        }
        self.children = None;
        self.n = 0;
    }
}

#[pyclass]
pub struct ODAC {
    confidence_level: f64,
    n_min: i64,
    tau: f64,
    
    root_node: ODACCluster,
    update_timer: i64,
    n_observations: i64,
    is_init: bool,
    structure_changed: bool,
}

#[pymethods]
impl ODAC {
    #[new]
    fn new(confidence_level: f64, n_min: i64, tau: f64) -> PyResult<Self> {
        if !(confidence_level > 0.0 && confidence_level < 1.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "confidence_level must be between 0 and 1."
            ));
        }
        
        if n_min <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_min must be greater than 0."
            ));
        }
        
        if tau <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tau must be greater than 0."
            ));
        }
        
        Ok(Self {
            confidence_level,
            n_min,
            tau,
            root_node: ODACCluster::new("ROOT".to_string()),
            update_timer: n_min,
            n_observations: 0,
            is_init: false,
            structure_changed: false,
        })
    }
    
    fn learn_one(&mut self, x: HashMap<String, f64>) {
        if x.is_empty() {
            return;
        }
        
        if self.structure_changed {
            self.structure_changed = false;
        }
        
        if !self.is_init {
            let ts_names: Vec<String> = x.keys().cloned().collect();
            self.root_node.initialize_timeseries(ts_names);
            self.structure_changed = true;
            self.is_init = true;
        }
        
        self.n_observations += 1;
        self.update_timer -= 1;
        
        // Update statistics for active clusters
        self.update_active_clusters(x);
        
        // Check for splits/merges periodically
        if self.update_timer == 0 {
            self.check_structure_changes();
            self.update_timer = self.n_min;
        }
    }
    
    fn predict_one(&self, x: HashMap<String, f64>) -> PyResult<usize> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "This algorithm does not predict anything. It builds a hierarchical cluster's structure."
        ))
    }
    
    fn get_n_clusters(&self) -> usize {
        self.count_clusters(&self.root_node)
    }
    
    fn get_n_active_clusters(&self) -> usize {
        self.count_active_clusters(&self.root_node)
    }
    
    fn get_height(&self) -> usize {
        self.calculate_height(&self.root_node)
    }
    
    fn get_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        summary.insert("n_clusters".to_string(), self.get_n_clusters());
        summary.insert("n_active_clusters".to_string(), self.get_n_active_clusters());
        summary.insert("height".to_string(), self.get_height());
        summary
    }
    
    fn render_ascii(&self, n_decimal_places: usize) -> PyResult<String> {
        if n_decimal_places == 0 || n_decimal_places >= 10 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_decimal_places must be between 1 and 9."
            ));
        }
        
        Ok(self.root_node.design_structure(n_decimal_places))
    }
    
    fn get_structure_changed(&self) -> bool {
        self.structure_changed
    }
}

impl ODAC {
    fn count_clusters(&self, node: &ODACCluster) -> usize {
        let mut count = 1;
        if let Some(ref children) = node.children {
            count += self.count_clusters(&children.first);
            count += self.count_clusters(&children.second);
        }
        count
    }
    
    fn count_active_clusters(&self, node: &ODACCluster) -> usize {
        if node.active {
            1
        } else if let Some(ref children) = node.children {
            self.count_active_clusters(&children.first) + self.count_active_clusters(&children.second)
        } else {
            0
        }
    }
    
    fn calculate_height(&self, node: &ODACCluster) -> usize {
        if let Some(ref children) = node.children {
            let height1 = self.calculate_height(&children.first);
            let height2 = self.calculate_height(&children.second);
            height1.max(height2) + 1
        } else {
            0
        }
    }
    
    fn find_all_active_clusters<'a>(&self, node: &'a ODACCluster) -> Vec<&'a ODACCluster> {
        let mut active_clusters = Vec::new();
        if node.active {
            active_clusters.push(node);
        } else if let Some(ref children) = node.children {
            active_clusters.extend(self.find_all_active_clusters(&children.first));
            active_clusters.extend(self.find_all_active_clusters(&children.second));
        }
        active_clusters
    }
    
    fn update_active_clusters(&mut self, x: HashMap<String, f64>) {
        // This is a simplified version - in a real implementation, we'd need to
        // properly traverse and update the tree structure
        if self.root_node.active {
            self.root_node.update_statistics(x);
        }
    }
    
    fn check_structure_changes(&mut self) {
        // This is a simplified version - in a real implementation, we'd need to
        // properly check all active clusters for split/merge conditions
        if self.root_node.active {
            self.root_node.calculate_coefficients(self.confidence_level);
            
            if self.root_node.test_split(self.tau) || self.root_node.test_aggregate() {
                self.structure_changed = true;
            }
        }
    }
}

#[pymodule]
fn odac_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ODAC>()?;
    m.add_class::<ODACCluster>()?;
    m.add_class::<ODACChildren>()?;
    m.add_class::<PearsonCorr>()?;
    m.add_class::<Var>()?;
    Ok(())
}
