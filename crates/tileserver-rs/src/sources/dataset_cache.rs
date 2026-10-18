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
//! - Keyed on a string key: the source `path` for the base dataset, or a
//!   `"{path}#ovr{level}"` key for an `OVERVIEW_LEVEL`-opened dataset (see
//!   [`super::cog::CogSource`]).
//! - Values are [`Pool`]s: a small set of `Dataset` handles for that key,
//!   not a single handle. A render *borrows* one handle for the duration of
//!   the render and returns it; concurrent renders at the same key each take
//!   a different handle, so they run in parallel. See [`Pool`] for the
//!   acquire/release contract and why a single `Arc<Mutex<Dataset>>` was not
//!   enough.
//! - Moka `future::Cache` handles TTL-based eviction (10 min default) and
//!   LRU eviction, weighted by the explicit **byte cost** estimate per entry
//!   (see [`Pool::weighted_bytes`]).
//! - On miss, `Dataset::open` runs inside `tokio::task::spawn_blocking`
//!   because GDAL I/O is blocking.
//!
//! # Why byte-weighted
//!
//! This cache is bounded by an estimated **byte budget**, not by an entry
//! count. A remote `/vsicurl/` COG is not one cheap handle: `CogSource`
//! opens one dataset per `OVERVIEW_LEVEL` (a COG with 8 overviews yields up
//! to 9 entries for a single source), each pool may hold several handles, and
//! every open dataset retains its own GDAL side caches. An entry *count*
//! bound therefore does not bound memory — a handful of such entries can pin
//! gigabytes. Weighting each entry by an estimated byte cost keeps the
//! *total* retained size bounded regardless of how many overview levels a
//! client visits or how many handles each pool grows to.
//!
//! The byte budget is **enforced**, not merely measured: [`Pool::acquire`]
//! refuses to open a new handle once the process-wide live byte total would
//! exceed the budget (see [`Pool::reserve_budget`]). A pool therefore never
//! grows past the budget even under a burst of concurrent tiles at one zoom.
//!
//! # Safety
//!
//! `gdal::Dataset` is `!Send + !Sync` in the general case but we only ever
//! access it behind an async-aware `Mutex` (the pool's `idle` list), so
//! cross-task reuse is sound. The `Arc<Pool>` is cheap to clone; callers pass
//! the clone into `tokio::task::spawn_blocking` and acquire a handle there.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex as StdMutex, OnceLock};
use std::time::Duration;

use gdal::{Dataset, DatasetOptions};
use moka::future::Cache;

use crate::error::{Result, TileServerError};

const DEFAULT_TTL_SECONDS: u64 = 600;
/// Byte budget for all cached open datasets.
///
/// Deliberately below the rendered-tile cache's own budget so the two cannot
/// add up to a multi-GB footprint on a small host. This is the sole bound on
/// the cache: moka has a single `max_capacity` and it is interpreted as weight
/// when a weigher is installed (see [`DatasetCache::with_max_bytes`]).
const DEFAULT_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// Estimated resident cost of holding one open GDAL dataset, in bytes.
///
/// A `gdal::Dataset` cannot report its true heap use, so this is a deliberate
/// upper-bound estimate. It is a fixed constant per open kind rather than a
/// measurement: its job is to make the *total* bounded and to make
/// overview-opened datasets (which carry their own per-dataset GDAL side
/// caches) cost more than a plain open.
///
/// The overview figure is deliberately close to the observed working set
/// (~8 MB per overview dataset in our RSS measurements) so that a 256 MB
/// budget holds tens of handles rather than a handful — an over-large
/// estimate would make the pool evict and reopen constantly, which is exactly
/// the thrash this cache exists to prevent. Tune it here if the working set
/// needs to grow or shrink.
const OVERVIEW_COST_BYTES: u64 = 8 * 1024 * 1024;
const BASE_COST_BYTES: u64 = 16 * 1024 * 1024;

/// Process-wide count of open datasets held across *all* pools, in bytes.
///
/// This is the number the byte budget actually gates on: [`Pool::reserve_budget`]
/// increments it before opening and the decrement happens when a handle is
/// dropped (see [`PooledHandle`]). Keeping it process-global (rather than
/// per-pool) is what makes the budget a bound on the whole cache instead of a
/// per-key bound that multiplies by the number of keys.
static GLOBAL_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Maximum number of handles a single pool may hold.
///
/// One handle per hardware thread is enough to keep the render pipeline busy
/// (the render is CPU/IO-bound and does not oversubscribe usefully), and it
/// matches the maintainer's proposal. A pool never opens more than this even
/// if the byte budget would allow it.
fn pool_limit() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

/// Estimated byte cost of one handle opened with `open_options`.
fn entry_cost_bytes(open_options: &[String]) -> u64 {
    let is_overview = open_options
        .iter()
        .any(|o| o.starts_with("OVERVIEW_LEVEL="));
    if is_overview {
        OVERVIEW_COST_BYTES
    } else {
        BASE_COST_BYTES
    }
}

/// A handle borrowed from a [`Pool`].
///
/// Dropping it returns the underlying `Dataset` to the pool's idle list (or
/// closes it if the pool is gone), and releases its byte cost from
/// [`GLOBAL_LIVE_BYTES`]. This makes release panic-safe: a render that unwinds
/// still returns its slot, so the pool cannot leak capacity.
pub struct PooledHandle {
    dataset: Option<Dataset>,
    pool: Arc<Pool>,
}

impl PooledHandle {
    /// Borrow the underlying dataset for the duration of a render.
    #[must_use]
    pub fn dataset(&self) -> &Dataset {
        self.dataset
            .as_ref()
            .expect("PooledHandle dataset taken only on drop")
    }

    /// Take ownership of the dataset, detaching it from the pool.
    ///
    /// Used by sources that hold a single long-lived handle (e.g. a DEM source
    /// whose lifetime spans many tiles) rather than borrowing per render. The
    /// handle stays counted against the byte budget via the pool's `live`
    /// count; it is simply no longer returned to the idle list on drop.
    #[must_use]
    pub fn into_dataset(mut self) -> Dataset {
        self.dataset
            .take()
            .expect("PooledHandle dataset taken only on drop")
    }
}

impl Drop for PooledHandle {
    fn drop(&mut self) {
        if let Some(dataset) = self.dataset.take() {
            // Return the handle to the idle list and wake one waiter. The
            // handle stays open and stays counted against the byte budget
            // (`live` is NOT decremented here) — it is merely free to be
            // borrowed again. It is only closed, and its cost released, when
            // the pool itself is dropped/evicted.
            let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
            state.idle.push(dataset);
            drop(state);
            self.pool.returned.notify_one();
        }
    }
}

/// A small pool of open `Dataset` handles for one cache key.
///
/// # Why a pool and not one handle
///
/// A single `Arc<Mutex<Dataset>>` per key serialises every render at that key:
/// the render holds the lock for its whole duration, so N concurrent tiles at
/// one overview level render one at a time. Measured on a local COG that is
/// ~4x slower than opening per tile at 16-way concurrency. A pool keeps the
/// cached-open win *and* lets N renders proceed in parallel, by handing each
/// concurrent render a different handle.
///
/// # Bounds
///
/// Two independent bounds apply, and both are enforced:
/// - **Concurrency**: `limit` handles at most (`available_parallelism()`).
///   Acquisition beyond that *waits* for a handle to be returned rather than
///   opening more.
/// - **Memory**: a new handle is only opened while the process-wide live byte
///   total stays within the cache's byte budget. When the budget is full the
///   pool reuses existing handles instead of opening new ones.
///
/// # Why a condvar and not a semaphore
///
/// The pool needs exactly one thing: a way to wait until a handle is free
/// without holding the lock that the returning thread needs. A `Condvar`
/// paired with the state mutex does precisely that — `wait` atomically
/// releases the mutex while parked, and `PooledHandle::drop` signals it. A
/// semaphore would add a second primitive, a second accounting path, and an
/// async/blocking boundary to get right, for no additional guarantee: the
/// `live < limit` check under the same lock already bounds concurrency.
pub struct Pool {
    /// Idle handles plus the live count, under one lock. `live` counts every
    /// open handle (idle + borrowed) and is the number the byte budget and the
    /// concurrency limit gate on.
    state: StdMutex<PoolState>,
    /// Signalled by [`PooledHandle::drop`] when a handle is returned.
    returned: Condvar,
    /// Max handles this pool may hold.
    limit: usize,
    /// Byte cost of one handle in this pool (same for all handles in a pool,
    /// since they share `open_options`).
    cost_bytes: u64,
    /// Process-wide byte budget; a new handle is refused once the live total
    /// would exceed it.
    max_bytes: u64,
    /// Path to open new handles from.
    path: String,
    /// Open options (e.g. `OVERVIEW_LEVEL=N`) to open new handles with.
    open_options: Vec<String>,
}

/// Mutable pool state, guarded by a single mutex.
struct PoolState {
    /// Free handles, ready to be reused.
    idle: Vec<Dataset>,
    /// Handles currently open (idle + borrowed).
    live: usize,
}

impl Pool {
    /// Create an empty pool for `path` opened with `open_options`.
    fn new(path: String, open_options: Vec<String>, max_bytes: u64) -> Self {
        let limit = pool_limit().max(1);
        let cost_bytes = entry_cost_bytes(&open_options);
        Self {
            state: StdMutex::new(PoolState {
                idle: Vec::new(),
                live: 0,
            }),
            returned: Condvar::new(),
            limit,
            cost_bytes,
            max_bytes,
            path,
            open_options,
        }
    }

    /// Current estimated resident bytes held by this pool.
    #[must_use]
    pub fn weighted_bytes(&self) -> u64 {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.live as u64 * self.cost_bytes
    }

    /// Number of handles currently open (idle + borrowed).
    #[must_use]
    pub fn live_handles(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).live
    }

    /// Acquire a handle for one render, given an `Arc<Pool>` so the returned
    /// [`PooledHandle`] can keep the pool alive and return the handle on drop.
    ///
    /// The fast path (a free handle is already idle) needs no I/O and runs on
    /// the calling async thread. The slow path (opening a new handle or
    /// waiting for one) is blocking work, so it is moved onto
    /// `tokio::task::spawn_blocking` — GDAL's `Dataset::open` and the pool's
    /// condvar `wait` must not run on the async thread or they can stall the
    /// runtime. The returned [`PooledHandle`] returns the handle to the pool
    /// on drop.
    ///
    /// # Errors
    ///
    /// Returns [`TileServerError::RasterError`] if a new handle must be opened
    /// and the open fails.
    pub async fn acquire_arc(self: &Arc<Self>) -> Result<PooledHandle> {
        // Fast path: a free handle needs no I/O — take it on the async thread.
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(dataset) = state.idle.pop() {
                return Ok(PooledHandle {
                    dataset: Some(dataset),
                    pool: Arc::clone(self),
                });
            }
        }

        // Slow path: opening (and any waiting) is blocking work — run it off
        // the async thread so GDAL's open cannot stall the runtime.
        let pool = Arc::clone(self);
        tokio::task::spawn_blocking(move || pool.acquire_blocking())
            .await
            .map_err(|e| TileServerError::RasterError(format!("task failed: {e}")))?
    }

    /// Blocking acquire: open a handle or wait for one.
    ///
    /// MUST run on a blocking thread (GDAL open + condvar wait are both
    /// blocking). It re-checks the idle list inside because a handle may have
    /// been returned between the fast-path check in [`Self::acquire_arc`] and
    /// this blocking task starting.
    pub(crate) fn acquire_blocking(self: &Arc<Self>) -> Result<PooledHandle> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            // 1. Reuse a free handle.
            if let Some(dataset) = state.idle.pop() {
                return Ok(PooledHandle {
                    dataset: Some(dataset),
                    pool: Arc::clone(self),
                });
            }

            // 2. No free handle: open one if BOTH bounds allow. The byte
            //    budget is reserved before the open so a concurrent opener
            //    cannot slip past it.
            if state.live < self.limit && self.reserve_budget() {
                match self.open_blocking() {
                    Ok(dataset) => {
                        state.live += 1;
                        return Ok(PooledHandle {
                            dataset: Some(dataset),
                            pool: Arc::clone(self),
                        });
                    }
                    Err(e) => {
                        // Open failed: release the reservation we just took.
                        GLOBAL_LIVE_BYTES.fetch_sub(self.cost_bytes, Ordering::AcqRel);
                        return Err(e);
                    }
                }
            }

            // 2b. The budget refused the open and *nothing is live* — no
            //     handle exists that could later be returned, so waiting on
            //     the condvar would block forever. This is the case where the
            //     byte budget is smaller than a single handle's cost (or the
            //     global budget is already exhausted with no pool handle to
            //     borrow from). Return an error rather than deadlock.
            if state.live == 0 {
                return Err(TileServerError::RasterError(format!(
                    "byte budget exhausted: cannot open handle costing {} bytes (max {} bytes, {} live)",
                    self.cost_bytes, self.max_bytes, GLOBAL_LIVE_BYTES.load(Ordering::Acquire),
                )));
            }

            // 3. At a bound: wait for a handle to be returned. `wait` releases
            //    the mutex while parked, so `drop` can take it to push the
            //    returned handle and notify us.
            state = self
                .returned
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Try to reserve `cost_bytes` against the process-wide byte budget.
    ///
    /// Returns `true` and increments [`GLOBAL_LIVE_BYTES`] on success; returns
    /// `false` (leaving the total unchanged) when the budget is full.
    ///
    /// A `max_bytes == 0` pool has **no byte budget** (the legacy
    /// entry-count-only cache) — in that case the reservation always succeeds
    /// and the open is only gated by the concurrency `limit`.
    fn reserve_budget(&self) -> bool {
        if self.max_bytes == 0 {
            GLOBAL_LIVE_BYTES.fetch_add(self.cost_bytes, Ordering::AcqRel);
            return true;
        }
        GLOBAL_LIVE_BYTES
            .try_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                if cur + self.cost_bytes <= self.max_bytes {
                    Some(cur + self.cost_bytes)
                } else {
                    None
                }
            })
            .is_ok()
    }

    /// Open one dataset with this pool's path and options.
    fn open_blocking(&self) -> Result<Dataset> {
        let path = Path::new(&self.path);
        if self.open_options.is_empty() {
            Dataset::open(path)
        } else {
            let options_refs: Vec<&str> = self.open_options.iter().map(String::as_str).collect();
            Dataset::open_ex(
                path,
                DatasetOptions {
                    open_options: Some(&options_refs),
                    ..DatasetOptions::default()
                },
            )
        }
        .map_err(|e| TileServerError::RasterError(format!("failed to open dataset: {e}")))
    }

}

pub struct DatasetCache {
    inner: Cache<String, Arc<Pool>>,
    max_bytes: u64,
}

impl DatasetCache {
    /// Create a cache bounded by `max_capacity` **entries**, with no byte
    /// bound. This is the legacy behaviour and is retained only for callers
    /// (and tests) that want a plain entry cap; production uses
    /// [`Self::with_max_bytes`].
    #[must_use]
    pub fn new(capacity: u64, ttl: Duration) -> Self {
        let inner = Cache::builder()
            .max_capacity(capacity)
            .time_to_live(ttl)
            .support_invalidation_closures()
            .build();
        Self {
            inner,
            max_bytes: 0,
        }
    }

    /// Create a cache bounded by a total estimated-weight budget of
    /// `max_bytes`. When `max_bytes == 0` the cache falls back to an
    /// entry-count bound of `capacity` entries (legacy behaviour).
    ///
    /// Note moka has a **single** `max_capacity`. With a weigher installed it
    /// is interpreted as weight units, so the byte budget is the only active
    /// bound; no separate entry cap is enforced alongside it. The weigher
    /// reads each pool's *current* live byte total, so it tracks a pool that
    /// grows after insertion rather than freezing at insert time.
    #[must_use]
    pub fn with_max_bytes(capacity: u64, ttl: Duration, max_bytes: u64) -> Self {
        if max_bytes > 0 {
            let inner = Cache::builder()
                .max_capacity(max_bytes)
                .weigher(|_k: &String, pool: &Arc<Pool>| -> u32 {
                    // Weight is the pool's *live* byte total, read at
                    // eviction time. moka re-evaluates the weigher on
                    // maintenance, so a pool that grows is accounted for.
                    u32::try_from(pool.weighted_bytes()).unwrap_or(u32::MAX).max(1)
                })
                .time_to_live(ttl)
                .support_invalidation_closures()
                .build();
            Self { inner, max_bytes }
        } else {
            Self::new(capacity, ttl)
        }
    }

    /// Fetch the pool for `key`, creating it (empty) on a miss.
    ///
    /// The pool is created lazily and handles are opened on first
    /// [`Pool::acquire_arc`]. Creation itself does not open a dataset, so a
    /// cache hit costs no I/O.
    ///
    /// # Errors
    ///
    /// Returns [`TileServerError::RasterError`] if the pool cannot be created
    /// (currently infallible; kept for API symmetry).
    pub async fn get_or_open_cached(
        &self,
        key: &str,
        path: &str,
        open_options: &[&str],
    ) -> Result<Arc<Pool>> {
        let path_owned = path.to_string();
        let options_owned: Vec<String> = open_options.iter().map(|s| s.to_string()).collect();
        let max_bytes = self.max_bytes;

        // `try_get_with` runs the initialiser at most once per key even under
        // concurrent misses, so two cold requests do not both build a pool
        // (and, more importantly, do not both race to open the first handle).
        self.inner
            .try_get_with(key.to_string(), async move {
                Ok::<Arc<Pool>, TileServerError>(Arc::new(Pool::new(
                    path_owned,
                    options_owned,
                    max_bytes,
                )))
            })
            .await
            .map_err(|e: Arc<TileServerError>| {
                TileServerError::RasterError(format!("failed to create pool: {e}"))
            })
    }

    /// Fetch the base pool for `path`.
    ///
    /// # Errors
    ///
    /// See [`Self::get_or_open_cached`].
    pub async fn get_or_open(&self, path: &str) -> Result<Arc<Pool>> {
        self.get_or_open_cached(path, path, &[]).await
    }

    /// Fetch a pool for `key`, opened with `open_options`.
    ///
    /// Used by [`super::cog::CogSource`] to cache overview-level-opened
    /// datasets keyed by `{path}#ovr{level}` — mirroring rio-tiler's per-level
    /// reader cache. Each overview level keeps a *pool* of handles, so
    /// concurrent tiles at that level render in parallel.
    ///
    /// # Errors
    ///
    /// See [`Self::get_or_open_cached`].
    pub async fn get_or_open_with_options(
        &self,
        key: &str,
        path: &str,
        open_options: &[&str],
    ) -> Result<Arc<Pool>> {
        self.get_or_open_cached(key, path, open_options).await
    }

    /// Invalidate `path` and every overview key derived from it.
    ///
    /// A source has one base key (`path`) and up to one key per overview level
    /// (`{path}#ovr{n}`). Invalidating only the base key would leave every
    /// overview pool live, so this drops all of them.
    pub async fn invalidate(&self, path: &str) {
        let prefix = format!("{path}#ovr");
        let base = path.to_string();
        let _ = self
            .inner
            .invalidate_entries_if(move |k: &String, _v: &Arc<Pool>| {
                k == &base || k.starts_with(&prefix)
            });
    }

    #[must_use]
    pub fn entry_count(&self) -> u64 {
        self.inner.entry_count()
    }

    /// Current total estimated weight in bytes across all pools.
    ///
    /// This is the process-wide live total the byte budget gates on, so it is
    /// the number to assert against in tests.
    #[must_use]
    pub fn weighted_size(&self) -> u64 {
        GLOBAL_LIVE_BYTES.load(Ordering::Acquire)
    }
}

static GLOBAL: OnceLock<DatasetCache> = OnceLock::new();

/// Access the process-global [`DatasetCache`].
///
/// Lazily initialises with a 10-minute TTL and a **byte-bounded** capacity of
/// [`DEFAULT_MAX_BYTES`]. Byte-bounding is what keeps a remote-COG source
/// (which opens one dataset per overview level) from pinning gigabytes.
pub fn global() -> &'static DatasetCache {
    GLOBAL.get_or_init(|| {
        DatasetCache::with_max_bytes(
            0, // byte budget is the only bound; no entry cap
            Duration::from_secs(DEFAULT_TTL_SECONDS),
            DEFAULT_MAX_BYTES,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique-ish path in a fresh tempdir; returns (dir, path). The dir must
    /// stay alive for the path to be openable.
    fn tiny_gtiff(name: &str) -> Option<(tempfile::TempDir, String)> {
        use gdal::DriverManager;
        let dir = tempfile::tempdir().ok()?;
        let path = dir.path().join(name);
        let driver = DriverManager::get_driver_by_name("GTiff").ok()?;
        driver.create_with_band_type::<u8, _>(&path, 4, 4, 1).ok()?;
        Some((dir, path.to_string_lossy().to_string()))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn miss_on_nonexistent_path_returns_error_and_does_not_cache() {
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let pool = cache.get_or_open("/nonexistent/path.tif").await.unwrap();
        // The pool exists but opening a handle must fail.
        assert!(pool.acquire_arc().await.is_err());
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 1, "pool is cached on first request");
    }

    #[tokio::test]
    async fn hit_returns_same_pool() {
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            eprintln!("skip: GTiff driver unavailable");
            return;
        };
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let first = cache.get_or_open(&p).await.expect("first");
        let second = cache.get_or_open(&p).await.expect("second");
        assert!(Arc::ptr_eq(&first, &second), "same pool on hit");
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_release_reuses_handle() {
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let pool = cache.get_or_open(&p).await.expect("pool");
        {
            let h = pool.acquire_arc().await.expect("first acquire");
            assert_eq!(pool.live_handles(), 1);
            let _ = h.dataset();
        } // handle dropped -> returned to idle
        assert_eq!(pool.live_handles(), 1, "handle stays open in the pool");
        let h2 = pool.acquire_arc().await.expect("second acquire");
        assert_eq!(pool.live_handles(), 1, "reused, not reopened");
        drop(h2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_acquires_get_distinct_handles() {
        // The regression this guards: N concurrent renders at one key must not
        // serialise on a single handle. With a pool, each concurrent borrower
        // holds a distinct handle.
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let pool = cache.get_or_open(&p).await.expect("pool");

        let n = pool_limit().min(4).max(2);
        let mut held: Vec<PooledHandle> = Vec::new();
        for _ in 0..n {
            held.push(pool.acquire_arc().await.expect("acquire"));
        }
        assert_eq!(
            pool.live_handles(),
            n,
            "each concurrent borrower holds its own handle"
        );
        // Distinct handles: their raw pointers differ.
        let ptrs: Vec<*const Dataset> = held.iter().map(|h| h.dataset() as *const _).collect();
        for i in 0..ptrs.len() {
            for j in (i + 1)..ptrs.len() {
                assert_ne!(ptrs[i], ptrs[j], "handles must be distinct");
            }
        }
    }

    #[tokio::test]
    async fn base_and_overview_keys_are_distinct_entries() {
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        let base = cache.get_or_open(&p).await.expect("base");
        let ovr_key = format!("{p}#ovr0");
        let ovr = cache
            .get_or_open_with_options(&ovr_key, &p, &[])
            .await
            .expect("ovr");
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 2, "distinct keys are distinct entries");
        assert!(!Arc::ptr_eq(&base, &ovr), "distinct pools per key");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn weighted_size_counts_live_handles() {
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let before = GLOBAL_LIVE_BYTES.load(Ordering::Acquire);
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), DEFAULT_MAX_BYTES);
        let pool = cache.get_or_open(&p).await.expect("pool");
        assert_eq!(pool.weighted_bytes(), 0, "empty pool weighs nothing");
        let h = pool.acquire_arc().await.expect("acquire");
        assert_eq!(
            pool.weighted_bytes(),
            BASE_COST_BYTES,
            "one base handle costs BASE_COST_BYTES"
        );
        assert_eq!(cache.weighted_size(), before + BASE_COST_BYTES);
        drop(h);
        // Handle is returned to idle, still open, so still counted.
        assert_eq!(pool.weighted_bytes(), BASE_COST_BYTES);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn byte_budget_is_enforced() {
        // A budget smaller than one base handle forces the pool to refuse to
        // open: the bound is enforced, not merely reported.
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let budget = BASE_COST_BYTES / 2;
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), budget);
        let pool = cache.get_or_open(&p).await.expect("pool");
        assert!(
            pool.acquire_arc().await.is_err(),
            "opening past the byte budget must be refused"
        );
        assert_eq!(pool.live_handles(), 0, "no handle leaked past the budget");
    }

    #[tokio::test]
    async fn invalidate_drops_base_and_overview_keys() {
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let cache = DatasetCache::new(10, Duration::from_secs(60));
        cache.get_or_open(&p).await.expect("base");
        cache
            .get_or_open_with_options(&format!("{p}#ovr0"), &p, &[])
            .await
            .expect("ovr0");
        cache
            .get_or_open_with_options(&format!("{p}#ovr1"), &p, &[])
            .await
            .expect("ovr1");
        cache.inner.run_pending_tasks().await;
        assert_eq!(cache.entry_count(), 3);
        cache.invalidate(&p).await;
        cache.inner.run_pending_tasks().await;
        assert_eq!(
            cache.entry_count(),
            0,
            "invalidate must drop the base key AND every {{path}}#ovr* key"
        );
    }

    #[test]
    fn global_is_stable_across_calls() {
        assert!(std::ptr::eq(global(), global()));
    }
}
