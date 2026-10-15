//! Process-global cache of open GDAL [`Dataset`] handles.
//!
//! # Why this exists
//!
//! Every call to `Dataset::open(path)` costs 10-50ms on a remote COG
//! because GDAL has to open the HTTP connection, fetch the TIFF header,
//! parse the IFDs, and populate the dataset-level metadata. The STAC
//! mosaic path currently opens one `Dataset` per asset per tile request,
//! so a 5-asset mosaic re-pays this cost on every tile served.
//!
//! Once opened, a `Dataset` can be safely reused across tile requests
//! (all state per-read is stack-local), so caching it eliminates that
//! latency on subsequent reads.
//!
//! # Design
//!
//! - Keyed on `path` (either a local filesystem path or a GDAL virtual
//!   path like `/vsicurl/https://...`).
//! - Values are `Arc<Mutex<Dataset>>`; the `Mutex` preserves the
//!   existing thread-safety contract used by [`super::cog::CogSource`].
//! - Moka `future::Cache` handles TTL-based eviction (10 min default)
//!   and LRU size-based eviction (100 entries default).
//! - On miss, `Dataset::open` runs inside `tokio::task::spawn_blocking`
//!   because GDAL I/O is blocking.
//!
//! # Safety
//!
//! `gdal::Dataset` is `!Send + !Sync` in the general case but we only
//! ever access it behind an async-aware `Mutex`, so cross-task
//! reuse is sound. The `Arc` is cheap to clone; callers pass the clone
//! into `tokio::task::spawn_blocking` and re-lock there.

use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use gdal::{Dataset, DatasetOptions};
use moka::future::Cache;
use tokio::sync::Mutex;

use crate::error::{Result, TileServerError};

const DEFAULT_TTL_SECONDS: u64 = 600;
const DEFAULT_CAPACITY: u64 = 100;

pub type CachedDataset = Arc<Mutex<Dataset>>;

pub struct DatasetCache {
    inner: Cache<String, CachedDataset>,
}

impl DatasetCache {
    #[must_use]
    pub fn new(capacity: u64, ttl: Duration) -> Self {
        Self {
            inner: Cache::builder()
                .max_capacity(capacity)
                .time_to_live(ttl)
                .build(),
        }
    }

    /// Fetch a cached [`Dataset`] for `path`, opening it on cache miss.
    ///
    /// The returned `Arc<Mutex<Dataset>>` can be cloned freely; the
    /// underlying `Dataset` is shared across all callers.
    ///
    /// # Errors
    ///
    /// Returns [`TileServerError::RasterError`] if `Dataset::open`
    /// fails or the blocking task panics.
    pub async fn get_or_open(&self, path: &str) -> Result<CachedDataset> {
        if let Some(cached) = self.inner.get(path).await {
            return Ok(cached);
        }
        let path_owned = path.to_string();
        let dataset = tokio::task::spawn_blocking(move || {
            Dataset::open(Path::new(&path_owned))
                .map_err(|e| TileServerError::RasterError(format!("failed to open dataset: {e}")))
        })
        .await
        .map_err(|e| TileServerError::RasterError(format!("task failed: {e}")))??;

        let arc = Arc::new(Mutex::new(dataset));
        self.inner.insert(path.to_string(), arc.clone()).await;
        Ok(arc)
    }

    /// Fetch a cached [`Dataset`] for `key`, opened with `open_options` on miss.
    ///
    /// Used by `CogSource` to cache overview-level-opened datasets keyed by
    /// (path, overview_level) — mirroring rio-tiler's per-level reader cache.
    /// Each overview level is opened **at most once per process**; the cached
    /// handle is then reused for every tile request that needs that level.
    /// The key must encode the open options (e.g. `{path}#ovr{level}`) so that
    /// different options don't collide in the cache.
    ///
    /// # Errors
    ///
    /// Returns [`TileServerError::RasterError`] if `Dataset::open_ex` fails or
    /// the blocking task panics. Failed opens do not populate the cache.
    pub async fn get_or_open_with_options(
        &self,
        key: &str,
        path: &str,
        open_options: &[&str],
    ) -> Result<CachedDataset> {
        if let Some(cached) = self.inner.get(key).await {
            return Ok(cached);
        }
        let path_owned = path.to_string();
        let options_owned: Vec<String> = open_options.iter().map(|s| s.to_string()).collect();
        let dataset = tokio::task::spawn_blocking(move || {
            let options_refs: Vec<&str> = options_owned.iter().map(String::as_str).collect();
            Dataset::open_ex(
                Path::new(&path_owned),
                DatasetOptions {
                    open_options: Some(&options_refs),
                    ..DatasetOptions::default()
                },
            )
            .map_err(|e| TileServerError::RasterError(format!("failed to open dataset: {e}")))
        })
        .await
        .map_err(|e| TileServerError::RasterError(format!("task failed: {e}")))??;

        let arc = Arc::new(Mutex::new(dataset));
        self.inner.insert(key.to_string(), arc.clone()).await;
        Ok(arc)
    }

    pub async fn invalidate(&self, path: &str) {
        self.inner.invalidate(path).await;
    }

    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.inner.entry_count()
    }
}

static GLOBAL: OnceLock<DatasetCache> = OnceLock::new();

/// Access the process-global [`DatasetCache`].
///
/// Lazily initialises with 100-entry capacity and 10-minute TTL. Future
/// work: allow config-driven tuning via `config.toml`.
pub fn global() -> &'static DatasetCache {
    GLOBAL.get_or_init(|| {
        DatasetCache::new(DEFAULT_CAPACITY, Duration::from_secs(DEFAULT_TTL_SECONDS))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn miss_on_nonexistent_path_returns_error_and_does_not_cache() {
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let path = "/nonexistent/path/will/fail/to/open.tif";
        let first = cache.get_or_open(path).await;
        assert!(first.is_err(), "first open must fail");
        assert_eq!(
            cache.entry_count(),
            0,
            "failed opens must not populate the cache"
        );
        let second = cache.get_or_open(path).await;
        assert!(second.is_err(), "second open must also fail");
        assert_eq!(cache.entry_count(), 0);
    }

    #[tokio::test]
    async fn miss_then_hit_returns_same_arc_when_gdal_can_open() {
        // Generate a tiny in-memory GeoTIFF via GDAL so the test runs
        // deterministically without a bundled fixture binary. Skips if
        // the MEM-to-GTiff driver path isn't available at runtime.
        use gdal::DriverManager;
        let tempdir = tempfile::tempdir().expect("create tempdir");
        let path = tempdir.path().join("tiny.tif");
        let Some(driver) = DriverManager::get_driver_by_name("GTiff").ok() else {
            eprintln!("skip: GTiff driver unavailable");
            return;
        };
        let created = driver
            .create_with_band_type::<u8, _>(&path, 4, 4, 1)
            .expect("create tiny tiff");
        drop(created);

        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let p = path.to_string_lossy().to_string();
        let first = cache.get_or_open(&p).await.expect("first open");
        let second = cache.get_or_open(&p).await.expect("second open");
        assert!(Arc::ptr_eq(&first, &second), "same Arc on hit");
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 1);
    }

    #[tokio::test]
    async fn invalidate_drops_entry() {
        use gdal::DriverManager;
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("tiny.tif");
        let Some(driver) = DriverManager::get_driver_by_name("GTiff").ok() else {
            return;
        };
        let _ = driver
            .create_with_band_type::<u8, _>(&path, 4, 4, 1)
            .expect("create tiny tiff");

        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let p = path.to_string_lossy().to_string();
        cache.get_or_open(&p).await.expect("cache populated");
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 1);
        cache.invalidate(&p).await;
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn global_is_stable_across_calls() {
        let a = global();
        let b = global();
        assert!(std::ptr::eq(a, b));
    }
}
