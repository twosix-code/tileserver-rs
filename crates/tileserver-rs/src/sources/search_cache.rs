//! Coalesced STAC bbox-search cache.
//!
//! A dynamic STAC source issues one `/search` per requested tile. A single
//! map viewport is tens of tiles sharing a near-identical WGS-84 bbox, so a
//! fast pan or zoom-out fires a burst of concurrent searches at the upstream
//! API. Real endpoints treat that as abuse and start refusing connections:
//! every tile in the viewport then stalls through its retries and the browser
//! reports the service as unreachable.
//!
//! This module collapses that burst. Searches are keyed by the exact search
//! parameters (api, collection, role, caps, rounded bbox), and:
//!
//! - **Single-flight**: concurrent callers for the same key await one in-flight
//!   upstream request instead of each issuing their own.
//! - **Short TTL**: a completed result is reused for a brief window, so a
//!   pan/zoom burst that straddles the request boundary still costs one call.
//!
//! The cache is process-global and bounded by entry count and TTL. It holds
//! only the discovered asset list (stable metadata), never tile pixels, so its
//! footprint is small and its entries are cheap to clone.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::broadcast;

use super::stac::StacAsset;
use crate::error::TileServerError;

/// How long a completed bbox search stays reusable.
///
/// Long enough to absorb the burst from one viewport settling (a pan/zoom
/// fires its tiles within ~1-2 s), short enough that a genuinely new area is
/// fetched on the next interaction. Ten seconds is deliberately modest: STAC
/// catalogues change slowly, and the cost of a rare stale refresh is one extra
/// upstream call, whereas the cost of no coalescing is the burst we are fixing.
const DEFAULT_TTL_SECONDS: u64 = 10;

/// Hard ceiling on cache entries. Keys are per-bbox; a long session across a
/// wide area would otherwise grow the map without bound. Tiny compared to the
/// tile cache — each entry is a handful of asset metadata records.
const DEFAULT_MAX_ENTRIES: usize = 512;

/// Result of an upstream bbox search: the asset list, or the error string the
/// first caller saw. Shared read-only across coalesced callers.
type SearchOutcome = Result<Vec<StacAsset>, String>;

/// One in-flight or completed search, shared by reference through the map.
///
/// `result` is `None` while the first caller is still awaiting the upstream
/// request; followers subscribe to `notify` and await the same completion. Once
/// filled, the entry serves as a plain TTL cache.
struct Entry {
    /// Broadcast channel used to wake followers when the first caller finishes.
    /// Each in-flight entry has exactly one sender (held by the map) and any
    /// number of receivers (the follower callers).
    notify: broadcast::Sender<()>,
    /// Filled once the first caller completes (success or error).
    result: Option<SearchOutcome>,
    /// When the entry was completed; drives TTL expiry. `None` while in flight.
    completed_at: Option<Instant>,
}

/// Process-global cache of coalesced STAC bbox searches.
pub struct SearchCache {
    ttl: Duration,
    max_entries: usize,
    entries: StdMutex<HashMap<String, Arc<StdMutex<Entry>>>>,
}

static GLOBAL: OnceLock<SearchCache> = OnceLock::new();

/// Access the process-global [`SearchCache`].
#[must_use]
pub fn global() -> &'static SearchCache {
    GLOBAL.get_or_init(|| {
        SearchCache::new(
            Duration::from_secs(DEFAULT_TTL_SECONDS),
            DEFAULT_MAX_ENTRIES,
        )
    })
}

impl SearchCache {
    fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries,
            entries: StdMutex::new(HashMap::new()),
        }
    }

    /// Return a cached/coalesced result for `key`, or run `search` to produce
    /// one.
    ///
    /// Only the **first** caller for a given key runs `search`; every caller
    /// that arrives while it is in flight awaits the same outcome. A completed
    /// outcome is served from cache until the TTL elapses.
    pub async fn get_or_search<F, Fut>(
        &self,
        key: String,
        search: F,
    ) -> std::result::Result<Vec<StacAsset>, TileServerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<Vec<StacAsset>, TileServerError>>,
    {
        // Decide the caller's role under the lock, then drop it before any
        // `await`: holding a std mutex across an await is a deadlock hazard.
        enum Role {
            Leader(Arc<StdMutex<Entry>>),
            Follower(broadcast::Receiver<()>, Arc<StdMutex<Entry>>),
            Cached(Vec<StacAsset>),
            CachedError(String),
        }

        let role = {
            let mut map = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            self.expire_locked(&mut map);

            match map.get(&key) {
                Some(entry) => {
                    let entry = Arc::clone(entry);
                    let guard = entry.lock().unwrap_or_else(|e| e.into_inner());
                    match &guard.result {
                        // Completed long enough ago to still be present: serve
                        // it (expiry above already dropped stale ones).
                        Some(Ok(assets)) => Role::Cached(assets.clone()),
                        Some(Err(msg)) => Role::CachedError(msg.clone()),
                        None => {
                            // In flight: subscribe before releasing the entry
                            // lock so a completion cannot be missed.
                            let rx = guard.notify.subscribe();
                            drop(guard);
                            Role::Follower(rx, entry)
                        }
                    }
                }
                None => {
                    // First caller: insert an in-flight entry and lead.
                    let (tx, _rx) = broadcast::channel(1);
                    let entry = Arc::new(StdMutex::new(Entry {
                        notify: tx,
                        result: None,
                        completed_at: None,
                    }));
                    map.insert(key.clone(), Arc::clone(&entry));
                    Role::Leader(entry)
                }
            }
        };

        match role {
            Role::Cached(assets) => Ok(assets),
            Role::CachedError(msg) => Err(TileServerError::StacError(msg)),
            Role::Follower(mut rx, entry) => {
                // Await the leader's completion signal. A `RecvError` means the
                // leader already finished and dropped the sender — the outcome
                // is in `entry.result`, so fall through and read it.
                let _ = rx.recv().await;
                let guard = entry.lock().unwrap_or_else(|e| e.into_inner());
                match &guard.result {
                    Some(Ok(assets)) => Ok(assets.clone()),
                    Some(Err(msg)) => Err(TileServerError::StacError(msg.clone())),
                    None => Err(TileServerError::StacError(
                        "coalesced stac search ended without a result".to_string(),
                    )),
                }
            }
            Role::Leader(entry) => {
                let outcome = search().await;
                let (notify, stored) = {
                    let mut guard = entry.lock().unwrap_or_else(|e| e.into_inner());
                    guard.result = Some(match &outcome {
                        Ok(assets) => Ok(assets.clone()),
                        Err(e) => Err(e.to_string()),
                    });
                    guard.completed_at = Some(Instant::now());
                    (guard.notify.clone(), guard.result.clone())
                };
                // Wake followers, then release the sender.
                let _ = notify.send(());
                drop(notify);
                match stored {
                    Some(Ok(assets)) => Ok(assets),
                    Some(Err(msg)) => Err(TileServerError::StacError(msg)),
                    None => unreachable!("leader just stored a result"),
                }
            }
        }
    }

    /// Drop completed entries whose TTL has elapsed. In-flight entries are
    /// never expired (they have no `completed_at`).
    fn expire_locked(&self, map: &mut HashMap<String, Arc<StdMutex<Entry>>>) {
        let now = Instant::now();
        map.retain(|_, entry| {
            let guard = entry.lock().unwrap_or_else(|e| e.into_inner());
            match guard.completed_at {
                Some(done) => now.duration_since(done) < self.ttl,
                None => true,
            }
        });

        // Enforce the entry ceiling. If still over after expiry, drop the
        // oldest completed entries; in-flight ones are retained regardless
        // (dropping them would orphan their followers).
        if map.len() > self.max_entries {
            let mut completed: Vec<(String, Instant)> = map
                .iter()
                .filter_map(|(k, v)| {
                    let guard = v.lock().unwrap_or_else(|e| e.into_inner());
                    guard.completed_at.map(|t| (k.clone(), t))
                })
                .collect();
            completed.sort_by_key(|(_, t)| *t);
            let excess = map.len() - self.max_entries;
            for (k, _) in completed.into_iter().take(excess) {
                map.remove(&k);
            }
        }
    }

    /// Number of cached (in-flight or completed) entries. Test helper.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the cache holds no entries. Test helper.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove every entry. Test helper, so tests do not observe each other.
    pub fn clear(&self) {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn asset(id: &str) -> StacAsset {
        StacAsset {
            id: id.to_string(),
            href: format!("https://example.invalid/{id}.tif"),
            bbox: [0.0, 0.0, 1.0, 1.0],
            title: id.to_string(),
            datetime: None,
            cloud_cover: None,
        }
    }

    #[tokio::test]
    async fn single_flight_collapses_concurrent_callers() {
        let cache = SearchCache::new(Duration::from_secs(10), 64);
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..16 {
            let cache_ref = &cache;
            let calls = Arc::clone(&calls);
            handles.push(async move {
                cache_ref
                    .get_or_search("k".to_string(), || async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        // Give followers time to join while this is in flight.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(vec![asset("a")])
                    })
                    .await
            });
        }
        let results = futures::future::join_all(handles).await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "16 concurrent callers must issue exactly one upstream search"
        );
        for r in results {
            assert_eq!(r.expect("all succeed").len(), 1);
        }
    }

    #[tokio::test]
    async fn completed_result_is_reused_within_ttl() {
        let cache = SearchCache::new(Duration::from_secs(10), 64);
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let calls = Arc::clone(&calls);
            cache
                .get_or_search("k".to_string(), || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![asset("a")])
                })
                .await
                .expect("ok");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "TTL reuse must serve all");
    }

    #[tokio::test]
    async fn expired_entry_is_refetched() {
        let cache = SearchCache::new(Duration::from_millis(1), 64);
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..2 {
            let calls = Arc::clone(&calls);
            cache
                .get_or_search("k".to_string(), || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![asset("a")])
                })
                .await
                .expect("ok");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "a stale entry must be refetched"
        );
    }

    #[tokio::test]
    async fn error_is_shared_and_not_retried_by_followers() {
        let cache = SearchCache::new(Duration::from_secs(10), 64);
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..5 {
            let cache_ref = &cache;
            let calls = Arc::clone(&calls);
            handles.push(async move {
                cache_ref
                    .get_or_search("k".to_string(), || async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        Err(TileServerError::StacError("boom".to_string()))
                    })
                    .await
            });
        }
        let results = futures::future::join_all(handles).await;

        assert_eq!(calls.load(Ordering::SeqCst), 1, "one upstream attempt");
        for r in results {
            assert!(r.is_err(), "every coalesced caller sees the error");
        }
    }

    #[tokio::test]
    async fn distinct_keys_are_independent() {
        let cache = SearchCache::new(Duration::from_secs(10), 64);
        let calls = Arc::new(AtomicUsize::new(0));

        for key in ["a", "b", "a", "c"] {
            let calls = Arc::clone(&calls);
            cache
                .get_or_search(key.to_string(), || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![asset(key)])
                })
                .await
                .expect("ok");
        }
        // a, b, c distinct; the second "a" is a TTL hit.
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn entry_ceiling_is_enforced() {
        let cache = SearchCache::new(Duration::from_secs(60), 4);
        // Insert 10 completed entries directly via the runtime-less path:
        // a zero-length future completed through a current-thread runtime.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for i in 0..10 {
            rt.block_on(async {
                cache
                    .get_or_search(format!("k{i}"), || async { Ok(vec![asset("a")]) })
                    .await
                    .expect("ok");
            });
        }
        assert!(
            cache.len() <= 4,
            "entry ceiling must cap the map, got {}",
            cache.len()
        );
    }
}
