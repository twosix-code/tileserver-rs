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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// Environment override for [`DEFAULT_MAX_BYTES`], in bytes.
///
/// A dynamic STAC mosaic opens one handle per asset, each costing
/// [`BASE_COST_BYTES`], so an N-asset mosaic needs roughly
/// `N * BASE_COST_BYTES` of budget to hold every asset at once. A 20-asset
/// supercell mosaic therefore saturates the 256 MB default exactly, and any
/// further asset fails with [`TileServerError::BudgetExhausted`] — tiles still
/// render (the mosaic degrades to partial coverage) but the log fills with
/// budget warnings and cold renders re-open assets constantly.
///
/// Deployments with large mosaics can raise this without a rebuild; the
/// default stays conservative so the crate does not assume a big host.
const MAX_BYTES_ENV: &str = "TILESERVER_DATASET_CACHE_MAX_BYTES";

/// Resolve the effective byte budget: env override if set and parseable,
/// otherwise [`DEFAULT_MAX_BYTES`].
///
/// An unparseable or zero value is ignored rather than treated as fatal — a
/// zero budget would make every cache miss fail, which is never the intent of
/// a typo in an env var.
fn resolve_max_bytes() -> u64 {
    match std::env::var(MAX_BYTES_ENV) {
        Ok(raw) => match raw.trim().parse::<u64>() {
            Ok(bytes) if bytes > 0 => bytes,
            _ => {
                tracing::warn!(
                    "{MAX_BYTES_ENV}={raw:?} is not a positive integer; using default {DEFAULT_MAX_BYTES}"
                );
                DEFAULT_MAX_BYTES
            }
        },
        Err(_) => DEFAULT_MAX_BYTES,
    }
}

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
/// closes it if the pool is gone). The handle's byte cost stays counted in the
/// pool's cache while it is borrowed, and is released only when the pool
/// itself is dropped/evicted (`Drop for Pool`). This makes release panic-safe:
/// a render that unwinds still returns its slot, so the pool cannot leak
/// capacity.
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

impl Drop for Pool {
    fn drop(&mut self) {
        // Release the byte budget held by any handles the pool still owns.
        //
        // This is the *last-resort* release: it only runs when the final
        // `Arc<Pool>` dies. `reclaim_idle` calls [`Pool::release_budget`]
        // directly, so a pool evicted from the cache while a caller still
        // holds an `Arc` refunds immediately instead of waiting for that
        // caller to finish. The `budget_released` flag makes whichever
        // happens first authoritative, so the counter is debited exactly
        // once no matter the order.
        let live = self
            .state
            .get_mut()
            .map_or_else(|e| e.into_inner().live, |s| s.live);
        self.release_budget(live);
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
    /// Cache-wide byte budget; a new handle is refused once the cache's live
    /// total would exceed it.
    max_bytes: u64,
    /// Shared live-bytes counter owned by the [`DatasetCache`] this pool
    /// belongs to. `reserve_budget` and `Drop` read/write it, so the budget is
    /// a per-cache bound rather than a process-wide one. Each pool holds a
    /// clone of the cache's `Arc`.
    live_bytes: Arc<AtomicU64>,
    /// Path to open new handles from.
    path: String,
    /// Open options (e.g. `OVERVIEW_LEVEL=N`) to open new handles with.
    open_options: Vec<String>,
    /// Set once this pool's bytes have been refunded to `live_bytes`.
    ///
    /// Refund must happen exactly once, and must NOT depend on `Arc` lifetime:
    /// `Drop for Pool` fires only when the last `Arc<Pool>` dies, but a caller
    /// can hold its own `Arc` (or a `PooledHandle`) long after the cache has
    /// evicted the pool. Tying the refund to `Drop` therefore defers — and in
    /// the reclaim path silently skips — the release, so the global counter
    /// stays inflated and the budget never frees. [`Pool::release_budget`]
    /// uses this flag so whoever releases first wins and the accounting is
    /// exact either way.
    budget_released: AtomicBool,
}

/// Mutable pool state, guarded by a single mutex.
struct PoolState {
    /// Free handles, ready to be reused.
    idle: Vec<Dataset>,
    /// Handles currently open (idle + borrowed).
    live: usize,
}

impl Pool {
    /// Create an empty pool for `path` opened with `open_options`, sharing the
    /// cache's live-bytes counter `live_bytes`.
    fn new(
        path: String,
        open_options: Vec<String>,
        max_bytes: u64,
        live_bytes: Arc<AtomicU64>,
    ) -> Self {
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
            live_bytes,
            path,
            open_options,
            budget_released: AtomicBool::new(false),
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

    /// Refund `live` handles worth of bytes to the shared counter — **once**.
    ///
    /// Both `Drop for Pool` (last `Arc` dying) and `DatasetCache::reclaim_idle`
    /// (eviction) call this. The `budget_released` flag makes the first caller
    /// authoritative so the counter is debited exactly once. Without the flag
    /// the two paths would double-refund, driving the counter *below* reality
    /// and letting the cache silently overshoot its budget.
    ///
    /// Takes `live` explicitly rather than reading the state lock: `Drop`
    /// already holds the state via `get_mut`, and locking there would deadlock
    /// a pool whose mutex is poisoned.
    fn release_budget(&self, live: usize) {
        if self.budget_released.swap(true, Ordering::AcqRel) {
            return;
        }
        self.live_bytes
            .fetch_sub(live as u64 * self.cost_bytes, Ordering::AcqRel);
    }

    /// Bytes this pool can hand back to the cache right now.
    ///
    /// This is the amount reclaim can free **without dropping a handle that is
    /// in use**. Two pools qualify, and the distinction was the defect:
    ///
    /// - **Fully idle** (`idle == live`): every handle is free, so the whole
    ///   pool is obsolete and its bytes (`live * cost`) can be refunded, after
    ///   which the pool is evicted and its handles closed.
    /// - **Partially idle** (`idle < live`, some borrowed): the borrowed
    ///   handles are mid-render and must stay counted, but the *idle* handles
    ///   are pure overhead — nobody is reading them, and closing them costs
    ///   nothing but a future reopen. Refunding just those bytes is safe and is
    ///   what makes the budget recyclable under a mosaic fill.
    ///
    /// Previously only the fully-idle case existed, so a mosaic that had every
    /// one of its pools mid-render (the normal shape of a dense multi-asset
    /// tile) presented nothing reclaimable and `acquire_with_reclaim` returned
    /// `BudgetExhausted` immediately — the saturation warning in the field.
    ///
    /// # Returns
    ///
    /// `(bytes_reclaimed, fully_idle)`. When `bytes_reclaimed == 0` the pool
    /// has no free handles and must not be evicted.
    ///
    /// # Contract with the caller
    ///
    /// For a **partially idle** pool the caller MUST close the returned count
    /// of idle handles ([`Self::take_idle_handles`]) — the refund is only
    /// honest once those handles are actually gone. For a **fully idle** pool
    /// the caller evicts the pool and the idle handles close with it.
    fn reclaimable_bytes(&self) -> (u64, bool) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let live = state.live;
        let idle = state.idle.len();
        if live == 0 || idle == 0 {
            return (0, false);
        }
        if idle == live {
            // Whole pool is dead weight: refund everything and let the caller
            // evict it.
            (live as u64 * self.cost_bytes, true)
        } else {
            // Only the free handles can be given back; the borrowed ones stay
            // counted in `live` and keep rendering.
            (idle as u64 * self.cost_bytes, false)
        }
    }

    /// Close and remove every idle handle, returning how many were taken.
    ///
    /// Used by the partially-idle reclaim path: the bytes for these handles
    /// were just refunded by [`Self::release_partial`], so they must actually
    /// be dropped for the accounting to match reality. Decorrelating the two
    /// would let the counter drift *below* the true footprint, which is how a
    /// cache silently overshoots its budget.
    fn take_idle_handles(&self) -> usize {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let taken = state.idle.len();
        // `Dataset` closes on drop; move the handles out before releasing the
        // lock so the (possibly slow) close does not serialise other renderers.
        let handles = std::mem::take(&mut state.idle);
        state.live -= taken;
        drop(state);
        drop(handles);
        taken
    }

    /// Refund `count` handles' worth of bytes to the shared counter.
    ///
    /// Unlike [`Self::release_budget`] this does **not** set
    /// `budget_released` — the pool stays in the cache with its borrowed
    /// handles still counted, so a later full-idle reclaim must still be able
    /// to refund the remainder.
    fn release_partial(&self, count: usize) {
        if count == 0 {
            return;
        }
        self.live_bytes
            .fetch_sub(count as u64 * self.cost_bytes, Ordering::AcqRel);
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
        // Budget-fit precheck. The per-pool `limit` (one handle per hardware
        // thread) is *not* derived from the byte budget, so a single pool can
        // ask for more bytes than the whole cache is allowed to hold. Left
        // unchecked that is not merely a slow path — it is a permanent
        // lockout: `reserve_budget` would refuse forever, and once this pool
        // holds nothing (`live == 0`) there is no handle to wait for, so every
        // later render returns an error. Refuse *before* taking the state lock
        // so the caller gets a precise, actionable message instead.
        if self.max_bytes > 0 && self.cost_bytes > self.max_bytes {
            return Err(TileServerError::BudgetExhausted {
                path: self.path.clone(),
                cost_bytes: self.cost_bytes,
                max_bytes: self.max_bytes,
                live_bytes: self.live_bytes.load(Ordering::Acquire),
            });
        }

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
                        self.live_bytes.fetch_sub(self.cost_bytes, Ordering::AcqRel);
                        return Err(e);
                    }
                }
            }

            // 2b. Nothing idle and the byte budget refused the open.
            //
            //     Two distinct cases, and conflating them is what turned a
            //     per-thread guard into a process-wide outage:
            //
            //     (i) `live > 0`. This pool holds handles that are merely
            //         borrowed. Waiting is correct: a render finishes, drops
            //         its `PooledHandle`, and notifies us.
            //
            //     (ii) `live == 0` and the cache's *global* total is
            //          genuinely occupied by **other** pools. Nothing this
            //          pool can wait for will ever free, so the condvar
            //          would park forever. The old code errored here, which
            //          bricked the server: a mosaic holding the whole budget
            //          made every other source fail permanently. Return the
            //          dedicated `BudgetExhausted` signal so the caller can
            //          reclaim from other pools and retry.
            if state.live == 0 {
                let live_total = self.live_bytes.load(Ordering::Acquire);
                // Drop the state lock first: reclaiming evicts *other* pools,
                // and their `Drop` takes their own state lock. Holding ours
                // across that is a lock-order hazard for no benefit.
                drop(state);
                return Err(TileServerError::BudgetExhausted {
                    path: self.path.clone(),
                    cost_bytes: self.cost_bytes,
                    max_bytes: self.max_bytes,
                    live_bytes: live_total,
                });
            }

            // 3. At a bound: wait for a handle to be returned. `wait` releases
            //    the mutex while parked, so `drop` can take it to push the
            //    returned handle and notify us.
            state = self.returned.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Try to reserve `cost_bytes` against the cache's byte budget.
    ///
    /// Returns `true` and increments the cache's live-bytes counter on
    /// success; returns `false` (leaving the total unchanged) when the budget
    /// is full.
    ///
    /// A `max_bytes == 0` pool has **no byte budget** (the legacy
    /// entry-count-only cache) — in that case the reservation always succeeds
    /// and the open is only gated by the concurrency `limit`.
    fn reserve_budget(&self) -> bool {
        if self.max_bytes == 0 {
            self.live_bytes.fetch_add(self.cost_bytes, Ordering::AcqRel);
            return true;
        }
        self.live_bytes
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
        .map_err(|e| {
            TileServerError::RasterError(format!(
                "failed to open {} {:?}: {e}",
                self.path, self.open_options
            ))
        })
    }
}

pub struct DatasetCache {
    inner: Cache<String, Arc<Pool>>,
    max_bytes: u64,
    /// This cache's live-bytes total (sum of `live * cost_bytes` over every
    /// pool it owns). Each pool holds a clone and reserves/releases against it
    /// via [`Pool::reserve_budget`] and `Drop for Pool`.
    live_bytes: Arc<AtomicU64>,
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
            live_bytes: Arc::new(AtomicU64::new(0)),
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
                    //
                    // A freshly created pool weighs 0 here, so moka sees
                    // weight 1 until the first open. That is fine: this
                    // weigher is only an eviction hint, and `reserve_budget`
                    // is the real bound on memory. Do not try to tune moka's
                    // capacity for memory — tune `DEFAULT_MAX_BYTES` instead.
                    u32::try_from(pool.weighted_bytes())
                        .unwrap_or(u32::MAX)
                        .max(1)
                })
                .time_to_live(ttl)
                .support_invalidation_closures()
                .build();
            Self {
                inner,
                max_bytes,
                live_bytes: Arc::new(AtomicU64::new(0)),
            }
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
        let live_bytes = Arc::clone(&self.live_bytes);

        // `try_get_with` runs the initialiser at most once per key even under
        // concurrent misses, so two cold requests do not both build a pool
        // (and, more importantly, do not both race to open the first handle).
        self.inner
            .try_get_with(key.to_string(), async move {
                Ok::<Arc<Pool>, TileServerError>(Arc::new(Pool::new(
                    path_owned,
                    options_owned,
                    max_bytes,
                    live_bytes,
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

    /// Acquire a handle from `pool`, reclaiming idle pools if the cache's byte
    /// budget is exhausted.
    ///
    /// [`Pool::acquire_blocking`] is synchronous and cannot reclaim on its own
    /// (reclamation awaits moka's maintenance). This wrapper is the async-side
    /// retry:
    ///
    /// 1. Fast path — try the acquire; on success, done.
    /// 2. On [`TileServerError::BudgetExhausted`], evict the cache's fully idle
    ///    pools and try once more. That is the common case in the field: one
    ///    STAC mosaic fills the budget with pools it is no longer rendering
    ///    from.
    /// 3. Still exhausted, or the budget cannot fit even one handle — surface
    ///    the error. It maps to `503`, not `500`: the condition is resource
    ///    pressure, not a defect in the request.
    ///
    /// Reclaim runs on the async thread, so the blocking acquire is spawned as
    /// usual and the retry happens between attempts rather than inside one.
    ///
    /// # Errors
    ///
    /// Returns the underlying raster error if the handle cannot be opened even
    /// after reclaiming.
    pub async fn acquire_with_reclaim(&self, pool: &Arc<Pool>) -> Result<PooledHandle> {
        // Note: this function cannot call `acquire_arc` and retry it directly,
        // because `PooledHandle` borrows no cache state — only the pool. That
        // is exactly what makes the retry safe: a failed acquire leaves the
        // pool unchanged, so a second attempt is a clean retry.
        match pool.acquire_arc().await {
            Ok(handle) => Ok(handle),
            Err(TileServerError::BudgetExhausted { .. }) => {
                let reclaimed = self.reclaim_idle().await;
                if reclaimed == 0 {
                    // Nothing idle to give back. Re-running the acquire would
                    // just reproduce the same error, so retry only when the
                    // reclaim actually freed budget.
                    return Err(TileServerError::BudgetExhausted {
                        path: pool.path.clone(),
                        cost_bytes: pool.cost_bytes,
                        max_bytes: pool.max_bytes,
                        live_bytes: self.live_bytes.load(Ordering::Acquire),
                    });
                }
                pool.acquire_arc().await
            }
            Err(e) => Err(e),
        }
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

    /// Whether a specific key is currently cached (used by observability to
    /// confirm a `{path}#ovr{n}` pool was created).
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Current total estimated weight in bytes across all pools in this cache.
    ///
    /// This is this cache's live total the byte budget gates on, so it is the
    /// number to assert against in tests.
    #[must_use]
    pub fn weighted_size(&self) -> u64 {
        self.live_bytes.load(Ordering::Acquire)
    }

    /// Evict every cached pool whose handles are all **idle**, and return how
    /// many bytes that reclaimed.
    ///
    /// This is the escape hatch for the budget lockout described on
    /// [`crate::error::TileServerError::BudgetExhausted`]: when one source (a
    /// STAC mosaic, typically) fills the whole budget with pools it is not
    /// currently rendering from, every *other* source is refused. Before this
    /// method existed the refusal was terminal and the process served 503s
    /// until it was restarted.
    ///
    /// Only **fully idle** pools are evicted. A pool with a currently borrowed
    /// handle is left alone: evicting it would orphan that handle's bytes (its
    /// `Drop` refunds `live * cost_bytes` including the borrowed handle, which
    /// then returns to a pool no longer in the cache). Idle pools are exactly
    /// the ones whose bytes no in-flight render depends on, so reclaiming them
    /// is safe.
    ///
    /// Called from the error path, so this must not block on a render. It only
    /// takes `try_lock`-free state locks (each pool's mutex is only ever held
    /// across a push/pop) and awaits moka's maintenance inline.
    pub async fn reclaim_idle(&self) -> u64 {
        // Snapshot candidates first: `iter()` borrows the cache, and we must
        // not hold that borrow across the awaits below.
        let candidates: Vec<(String, Arc<Pool>)> =
            self.inner.iter().map(|(k, v)| (k.to_string(), v)).collect();

        let mut reclaimed = 0u64;
        for (key, pool) in candidates {
            let (bytes, fully_idle) = pool.reclaimable_bytes();
            if bytes == 0 {
                continue;
            }
            if fully_idle {
                // Whole pool is dead weight: refund all of it, then evict.
                // The refund is explicit rather than left to `Drop for Pool`
                // because this loop's snapshot and any caller-held `Arc<Pool>`
                // keep the pool alive after eviction — so `Drop` would not run
                // here and the budget would stay inflated (the original defect).
                let live = pool.live_handles();
                pool.release_budget(live);
                self.inner.invalidate(&key).await;
                drop(pool);
                reclaimed = reclaimed.saturating_add(bytes);
            } else {
                // Partially idle: some handles are mid-render and must stay
                // counted, but the free ones are pure overhead. Refund exactly
                // those and close the handles so the accounting matches
                // reality. The pool stays cached — its borrowed handles will
                // come back to it.
                let freed = bytes / pool.cost_bytes;
                let taken = pool.take_idle_handles();
                debug_assert_eq!(
                    usize::try_from(freed).unwrap_or(usize::MAX),
                    taken,
                    "refund must match handles actually closed"
                );
                pool.release_partial(taken);
                reclaimed = reclaimed.saturating_add(bytes);
            }
        }

        if reclaimed > 0 {
            // Let moka finish its eviction bookkeeping now that the pools are
            // actually freed.
            self.inner.run_pending_tasks().await;
        }
        reclaimed
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
        let max_bytes = resolve_max_bytes();
        DatasetCache::with_max_bytes(
            0, // byte budget is the only bound; no entry cap
            Duration::from_secs(DEFAULT_TTL_SECONDS),
            max_bytes,
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
    async fn failed_open_keeps_empty_pool() {
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

        let n = pool_limit().clamp(2, 4);
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
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), DEFAULT_MAX_BYTES);
        let pool = cache.get_or_open(&p).await.expect("pool");
        assert_eq!(pool.weighted_bytes(), 0, "empty pool weighs nothing");
        let h = pool.acquire_arc().await.expect("acquire");
        assert_eq!(
            pool.weighted_bytes(),
            BASE_COST_BYTES,
            "one base handle costs BASE_COST_BYTES"
        );
        assert_eq!(cache.weighted_size(), BASE_COST_BYTES);
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
            matches!(
                pool.acquire_arc().await,
                Err(TileServerError::BudgetExhausted { .. })
            ),
            "opening past the byte budget must be refused with BudgetExhausted"
        );
        assert_eq!(pool.live_handles(), 0, "no handle leaked past the budget");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn budget_too_small_for_one_handle_is_still_recoverable() {
        // When one handle costs more than the whole budget the pool can never
        // open and `live` stays 0 forever, so the caller is refused forever.
        // That is a misconfiguration, and it must still surface as
        // `BudgetExhausted` rather than a generic raster error: the reclaim
        // retry then finds nothing to free and reports the same signal, so the
        // operator sees one consistent, actionable message (503, not 500).
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let budget = BASE_COST_BYTES - 1;
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), budget);
        let pool = cache.get_or_open(&p).await.expect("pool");
        let err = match pool.acquire_arc().await {
            Ok(_) => panic!("must be refused"),
            Err(e) => e,
        };
        assert!(
            matches!(err, TileServerError::BudgetExhausted { .. }),
            "an unfittable handle must report BudgetExhausted, got: {err}"
        );
        // Reclaiming cannot help — the budget is not held by anyone else.
        assert_eq!(cache.reclaim_idle().await, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reclaim_idle_frees_budget_for_another_source() {
        // Regression for the production lockout: one source fills the whole
        // byte budget with pools it is no longer rendering from, and every
        // other source is refused forever. `reclaim_idle` must hand the
        // budget back so the second source can open.
        let Some((_dir1, p1)) = tiny_gtiff("first.tif") else {
            return;
        };
        let Some((_dir2, p2)) = tiny_gtiff("second.tif") else {
            return;
        };
        // Budget fits exactly one base handle.
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), BASE_COST_BYTES);

        let pool1 = cache.get_or_open(&p1).await.expect("pool1");
        {
            // Hold and release, so pool1 owns one *idle* handle and its bytes.
            let _h = pool1.acquire_arc().await.expect("acquire pool1");
        }
        assert_eq!(cache.weighted_size(), BASE_COST_BYTES, "budget is full");

        // A second source cannot open: the budget is entirely pool1's.
        let pool2 = cache.get_or_open(&p2).await.expect("pool2");
        assert!(
            matches!(
                pool2.acquire_arc().await,
                Err(TileServerError::BudgetExhausted { .. })
            ),
            "second source must be refused while the budget is full"
        );

        // Reclaiming the idle pool frees the budget...
        let reclaimed = cache.reclaim_idle().await;
        assert_eq!(reclaimed, BASE_COST_BYTES, "idle pool's bytes are freed");
        assert_eq!(cache.weighted_size(), 0, "budget is released");

        // ...and the second source now opens. This is the behaviour the old
        // hard-error path made impossible: the server was stuck on 500s until
        // restarted.
        let handle2 = pool2.acquire_arc().await.expect("acquire after reclaim");
        assert_eq!(cache.weighted_size(), BASE_COST_BYTES);
        drop(handle2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reclaim_idle_skips_pools_with_borrowed_handles() {
        // Evicting a pool with a borrowed handle would orphan that handle: the
        // pool's `Drop` refunds `live * cost_bytes` (including the borrowed
        // one), and the handle then returns to a pool no longer in the cache.
        // `reclaim_idle` must therefore leave busy pools alone.
        let Some((_dir, p)) = tiny_gtiff("busy.tif") else {
            return;
        };
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), BASE_COST_BYTES);
        let pool = cache.get_or_open(&p).await.expect("pool");

        let held = pool.acquire_arc().await.expect("acquire");
        assert_eq!(cache.reclaim_idle().await, 0, "busy pool is not evicted");
        assert_eq!(cache.weighted_size(), BASE_COST_BYTES, "budget untouched");

        drop(held);
        // Now fully idle, so it becomes reclaimable.
        assert_eq!(cache.reclaim_idle().await, BASE_COST_BYTES);
        assert_eq!(cache.weighted_size(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reclaim_frees_partially_idle_pool_without_evicting_it() {
        // The field saturation this fixes: a dynamic STAC mosaic holds every
        // pool mid-render, each with one handle borrowed and one idle. Under
        // the old fully-idle-only rule nothing was reclaimable, so
        // `acquire_with_reclaim` re-raised `BudgetExhausted` immediately and
        // the log filled with budget warnings while coverage degraded.
        //
        // A partially idle pool must instead hand back the *free* handles'
        // bytes while keeping the borrowed handles counted and the pool cached.
        let Some((_dir1, p1)) = tiny_gtiff("busy1.tif") else {
            return;
        };
        let Some((_dir2, p2)) = tiny_gtiff("waiting.tif") else {
            return;
        };
        // Budget fits exactly two base handles.
        let budget = BASE_COST_BYTES * 2;
        let cache = DatasetCache::with_max_bytes(0, Duration::from_secs(60), budget);

        let pool1 = cache.get_or_open(&p1).await.expect("pool1");
        // Two handles live: one idle, one borrowed.
        let held1 = pool1.acquire_arc().await.expect("first handle");
        let idle1 = pool1.acquire_arc().await.expect("second handle");
        drop(idle1); // return to the idle list, bytes still counted
        assert_eq!(pool1.live_handles(), 2);
        assert_eq!(cache.weighted_size(), budget, "budget is full");

        // A second source is refused while pool1 holds both handles.
        let pool2 = cache.get_or_open(&p2).await.expect("pool2");
        assert!(
            matches!(
                pool2.acquire_arc().await,
                Err(TileServerError::BudgetExhausted { .. })
            ),
            "second source must be refused while the budget is full"
        );

        // Reclaim: pool1 is partially idle (1 of 2 free), so exactly one
        // handle's worth of bytes comes back and the borrowed one stays.
        let reclaimed = cache.reclaim_idle().await;
        assert_eq!(reclaimed, BASE_COST_BYTES, "only the idle handle is freed");
        assert_eq!(
            cache.weighted_size(),
            BASE_COST_BYTES,
            "the borrowed handle is still counted"
        );
        assert_eq!(pool1.live_handles(), 1, "the idle handle was closed");
        assert!(
            cache.contains_key(&p1),
            "a partially idle pool must stay cached, not be evicted"
        );

        // The freed bytes are enough for the blocked source.
        let handle2 = pool2.acquire_arc().await.expect("acquire after reclaim");

        // The render that borrowed pool1's handle still holds a valid dataset.
        assert!(held1.dataset().raster_size().0 > 0, "borrowed handle alive");

        drop(handle2);
        drop(held1);
        // Both pools are now fully idle: pool1 holds one handle, pool2 one.
        // One reclaim pass frees both.
        assert_eq!(cache.reclaim_idle().await, BASE_COST_BYTES * 2);
        assert!(!cache.contains_key(&p1), "fully idle pool is evicted");
        assert!(!cache.contains_key(&p2), "fully idle pool is evicted");
        assert_eq!(cache.weighted_size(), 0);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evicted_pool_releases_its_bytes() {
        // Regression: `Pool` had no `Drop`, so an evicted pool (TTL or
        // `invalidate`) closed its idle handles but left `live * cost_bytes`
        // in the cache's live-bytes counter. The budget then filled up and
        // stayed full, and the server stopped opening datasets after a while.
        //
        // This test exercises `Drop for Pool` directly (a pool with an idle
        // handle), with its own standalone live-bytes counter, so that
        // dropping the last `Arc<Pool>` fires `Drop` deterministically — no
        // dependence on moka's *async* eviction timing, which is deliberately
        // out of scope here. The assertions are against this counter, so there
        // is no process-global state to race — no `#[serial]` needed.
        let Some((_dir, p)) = tiny_gtiff("tiny.tif") else {
            return;
        };
        let live_bytes = Arc::new(AtomicU64::new(0));
        let pool = Arc::new(Pool::new(
            p.clone(),
            vec![],
            DEFAULT_MAX_BYTES,
            Arc::clone(&live_bytes),
        ));
        {
            let _h = pool.acquire_arc().await.expect("acquire");
            assert_eq!(
                live_bytes.load(Ordering::Acquire),
                BASE_COST_BYTES,
                "one live handle is counted"
            );
        } // handle dropped -> returned to idle, still counted
        assert_eq!(
            live_bytes.load(Ordering::Acquire),
            BASE_COST_BYTES,
            "idle handle stays counted while the pool is alive"
        );

        // Drop the only strong `Arc<Pool>`. `Drop for Pool` must release the
        // idle handle's bytes.
        drop(pool);
        assert_eq!(
            live_bytes.load(Ordering::Acquire),
            0,
            "dropping the pool must release the idle handle's bytes"
        );
    }

    #[test]
    fn global_is_stable_across_calls() {
        assert!(std::ptr::eq(global(), global()));
    }

    /// `resolve_max_bytes` must prefer a valid env override, and must fall
    /// back to the default for anything unparseable or non-positive.
    ///
    /// Run in one test to avoid env races between parallel test threads.
    #[test]
    fn resolve_max_bytes_honours_env_override() {
        // SAFETY: `set_var`/`remove_var` are unsafe in edition 2024 because a
        // concurrent `getenv` in another thread could observe a torn value.
        // This test owns `MAX_BYTES_ENV` exclusively — no other test reads or
        // writes it (it is not used by production code, only by
        // `resolve_max_bytes` at cache init), and we restore the previous
        // value before returning so the process environment is unchanged.
        unsafe {
            let prev = std::env::var(MAX_BYTES_ENV).ok();

            std::env::set_var(MAX_BYTES_ENV, "1073741824");
            assert_eq!(resolve_max_bytes(), 1024 * 1024 * 1024);

            std::env::set_var(MAX_BYTES_ENV, "  268435456  ");
            assert_eq!(resolve_max_bytes(), 256 * 1024 * 1024, "whitespace trimmed");

            std::env::set_var(MAX_BYTES_ENV, "0");
            assert_eq!(resolve_max_bytes(), DEFAULT_MAX_BYTES, "zero is rejected");

            std::env::set_var(MAX_BYTES_ENV, "not-a-number");
            assert_eq!(resolve_max_bytes(), DEFAULT_MAX_BYTES, "garbage rejected");

            std::env::remove_var(MAX_BYTES_ENV);
            assert_eq!(resolve_max_bytes(), DEFAULT_MAX_BYTES, "unset uses default");

            match prev {
                Some(v) => std::env::set_var(MAX_BYTES_ENV, v),
                None => std::env::remove_var(MAX_BYTES_ENV),
            }
        }
    }
}
