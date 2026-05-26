use pyo3::prelude::*;

mod bico;
mod clustream;
mod dbstream;
mod denstream;
mod kmeans;
mod odac;
mod streamkmeans;

use bico::BICO;
use clustream::{CluStream, CluStreamMicroCluster};
use dbstream::{DBStream, DBStreamMicroCluster};
use denstream::{DenStream, DenStreamMicroCluster};
use kmeans::KMeans;
use odac::{ODAC, ODACCluster, ODACChildren, PearsonCorr, Var};
use streamkmeans::StreamKMeans;

/// A Python module implemented in Rust for stream clustering algorithms.
#[pymodule]
fn stream_clustering(_py: Python, m: &PyModule) -> PyResult<()> {
    
    // BICO algorithm:
    m.add_class::<BICO>()?;
    
    // CluStream algorithm:
    m.add_class::<CluStream>()?;
    m.add_class::<CluStreamMicroCluster>()?;
    
    // DBStream algorithm:
    m.add_class::<DBStream>()?;
    m.add_class::<DBStreamMicroCluster>()?;
    
    // DenStream algorithm:
    m.add_class::<DenStream>()?;
    m.add_class::<DenStreamMicroCluster>()?;
    
    // KMeans algorithm:
    m.add_class::<KMeans>()?;
    
    // ODAC algorithm:
    m.add_class::<ODAC>()?;
    m.add_class::<ODACCluster>()?;
    m.add_class::<ODACChildren>()?;
    m.add_class::<PearsonCorr>()?;
    m.add_class::<Var>()?;
    
    // StreamKMeans algorithm:
    m.add_class::<StreamKMeans>()?;
    Ok(())
}
