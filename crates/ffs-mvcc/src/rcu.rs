//! Read-Copy-Update (RCU) primitives for lock-free metadata reads.
//!
//! This module provides RCU-style data structures backed by [`arc_swap::ArcSwap`]
//! for truly lock-free reader access with no atomic increments on the reader path.
//!
//! # Design
//!
//! - **Readers**: Call [`RcuCell::load`] to get a [`Guard`] — a zero-cost handle
//!   that borrows the current value without any atomic increment. Multiple
//!   concurrent readers never block each other or writers.
//!
//! - **Writers**: Call [`RcuCell::update`] to atomically publish a new value.
//!   Writers coordinate externally (e.g., via `Mutex`) if needed. The old
//!   value is freed through `Arc` reference counting once all readers finish.
//!
//! - **Reclamation (QSBR)**: Old values are reclaimed when the last `Guard`
//!   or `Arc` reference is dropped. The `arc-swap` crate internally uses a
//!   debt-based scheme that tracks quiescent states — once all reader guards
//!   from a given generation are dropped, the old `Arc` can be freed.
//!
//! # `unsafe_code = "forbid"` Compliance
//!
//! All RCU operations are safe Rust. The `arc-swap` crate encapsulates the
//! necessary atomics internally while exposing an entirely safe API.
//!
//! # Logging
//!
//! - **TRACE** `ffs::mvcc::rcu`: `rcu_cell_load` — reader load (guard created)
//! - **DEBUG** `ffs::mvcc::rcu`: `rcu_cell_update` — writer publishes new value
//! - **INFO**  `ffs::mvcc::rcu`: `rcu_map_update` — map entry updated or inserted
//! - **WARN**  `ffs::mvcc::rcu`: `rcu_map_high_churn` — map updates exceeding churn threshold
//! - **ERROR** `ffs::mvcc::rcu`: `rcu_map_inconsistency` — map invariant violation

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, trace, warn};

// ─── RcuCell ────────────────────────────────────────────────────────────────

/// A single RCU-protected value.
///
/// Provides lock-free reads (no atomic increments on the fast path) and
/// atomic updates via [`ArcSwap`]. Multiple concurrent readers never block
/// each other or writers.
///
/// # Examples
///
/// ```
/// use ffs_mvcc::rcu::RcuCell;
///
/// let cell = RcuCell::new(42_u64);
///
/// // Reader path — no locks, no atomic increments
/// let value = cell.load();
/// assert_eq!(**value, 42);
///
/// // Writer path — atomic swap
/// cell.update(100);
/// assert_eq!(**cell.load(), 100);
/// ```
pub struct RcuCell<T> {
    inner: ArcSwap<T>,
    update_count: AtomicU64,
}

impl<T: fmt::Debug> fmt::Debug for RcuCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RcuCell")
            .field("value", &*self.load_arc())
            .field("update_count", &self.update_count())
            .finish()
    }
}

impl<T> RcuCell<T> {
    /// Create a new `RcuCell` with an initial value.
    pub fn new(value: T) -> Self {
        Self {
            inner: ArcSwap::from_pointee(value),
            update_count: AtomicU64::new(0),
        }
    }

    /// Create from an existing `Arc<T>`.
    pub fn from_arc(arc: Arc<T>) -> Self {
        Self {
            inner: ArcSwap::from(arc),
            update_count: AtomicU64::new(0),
        }
    }

    /// Load the current value without any atomic increment.
    ///
    /// Returns a [`arc_swap::Guard`] that borrows the current `Arc<T>`.
    /// The guard must not be held across yield points or long operations
    /// — prefer short-lived reads.
    ///
    /// This is the primary reader API and is completely lock-free.
    #[inline]
    pub fn load(&self) -> arc_swap::Guard<Arc<T>> {
        let guard = self.inner.load();
        trace!(
            target: "ffs::mvcc::rcu",
            update_count = self.update_count.load(Ordering::Relaxed),
            "rcu_cell_load"
        );
        guard
    }

    /// Load the current value as a full `Arc<T>`.
    ///
    /// Unlike [`load`](Self::load), this performs an atomic increment on the
    /// `Arc` reference count. Use when you need to hold the value beyond
    /// the scope of a guard (e.g., across async yield points).
    #[inline]
    pub fn load_arc(&self) -> Arc<T> {
        self.inner.load_full()
    }

    /// Atomically publish a new value.
    ///
    /// All subsequent reads see the new value. Readers that loaded the
    /// old value before this call continue to see it until they release
    /// their guard/arc. The old value is freed when the last reference
    /// drops (QSBR-like reclamation via `Arc` refcount).
    pub fn update(&self, new_value: T) {
        self.inner.store(Arc::new(new_value));
        let count = self.update_count.fetch_add(1, Ordering::Relaxed) + 1;
        debug!(
            target: "ffs::mvcc::rcu",
            update_count = count,
            "rcu_cell_update"
        );
    }

    /// Atomically publish a new value from an existing `Arc<T>`.
    pub fn update_arc(&self, new_arc: Arc<T>) {
        self.inner.store(new_arc);
        let count = self.update_count.fetch_add(1, Ordering::Relaxed) + 1;
        debug!(
            target: "ffs::mvcc::rcu",
            update_count = count,
            "rcu_cell_update"
        );
    }

    /// Swap the current value, returning the old one.
    ///
    /// The caller receives the old `Arc<T>`; if no other readers hold it,
    /// the value is freed when this `Arc` drops.
    pub fn swap(&self, new_value: T) -> Arc<T> {
        let old = self.inner.swap(Arc::new(new_value));
        self.update_count.fetch_add(1, Ordering::Relaxed);
        old
    }

    /// Number of updates performed since creation.
    #[must_use]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

// ─── RcuMap ─────────────────────────────────────────────────────────────────

/// An RCU-protected immutable map for metadata caching.
///
/// The map is stored as `ArcSwap<BTreeMap<K, Arc<V>>>`. Readers get a
/// lock-free snapshot of the entire map. Writers produce a new map version
/// (copy-on-write) and publish it atomically.
///
/// This is designed for metadata caches where:
/// - Reads vastly outnumber writes (stat, readdir, lookup)
/// - The map is small-to-moderate (hundreds to thousands of entries)
/// - Writes are infrequent (inode creation, directory modification)
///
/// For large maps with frequent writes, consider [`ShardedMvccStore`](crate::sharded::ShardedMvccStore).
///
/// # Examples
///
/// ```
/// use ffs_mvcc::rcu::RcuMap;
///
/// let map: RcuMap<u64, String> = RcuMap::new();
///
/// // Writer: insert a new entry (copy-on-write)
/// map.insert(42, "hello".to_string());
///
/// // Reader: lock-free lookup
/// let snapshot = map.load();
/// assert_eq!(snapshot.get(&42).map(|v| v.as_str()), Some("hello"));
/// ```
pub struct RcuMap<K, V> {
    inner: ArcSwap<BTreeMap<K, Arc<V>>>,
    /// Write-side mutex: serializes COW updates to the map.
    /// Readers never acquire this — only writers.
    write_lock: Mutex<()>,
    update_count: AtomicU64,
    /// Churn threshold: log a warning if updates exceed this rate.
    churn_threshold: u64,
}

impl<K: fmt::Debug + Ord, V: fmt::Debug> fmt::Debug for RcuMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snap = self.inner.load_full();
        f.debug_struct("RcuMap")
            .field("entry_count", &snap.len())
            .field("update_count", &self.update_count.load(Ordering::Relaxed))
            .field("churn_threshold", &self.churn_threshold)
            .finish_non_exhaustive()
    }
}

impl<K: Clone + Ord + Hash + fmt::Debug, V: Clone + fmt::Debug> RcuMap<K, V> {
    /// Create a new empty RCU-protected map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(BTreeMap::new()),
            write_lock: Mutex::new(()),
            update_count: AtomicU64::new(0),
            churn_threshold: 10_000,
        }
    }

    /// Create with a custom churn warning threshold.
    #[must_use]
    pub fn with_churn_threshold(churn_threshold: u64) -> Self {
        Self {
            churn_threshold,
            ..Self::new()
        }
    }

    /// Load a lock-free snapshot of the map.
    ///
    /// Returns a guard that borrows the current map. No locks, no atomic
    /// increments on the reader fast path.
    #[inline]
    pub fn load(&self) -> arc_swap::Guard<Arc<BTreeMap<K, Arc<V>>>> {
        self.inner.load()
    }

    /// Load the current map as a full `Arc`.
    #[inline]
    pub fn load_arc(&self) -> Arc<BTreeMap<K, Arc<V>>> {
        self.inner.load_full()
    }

    /// Look up a key, returning a cloned `Arc<V>` if present.
    ///
    /// This is a convenience method that loads the map and looks up
    /// the key in a single operation.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        let snap = self.inner.load();
        snap.get(key).cloned()
    }

    /// Insert or update a key-value pair (copy-on-write).
    ///
    /// Acquires the write-side mutex, clones the current map, inserts the
    /// entry, and publishes the new version atomically. Readers never block.
    pub fn insert(&self, key: K, value: V) {
        let guard = self.write_lock.lock();
        let old = self.inner.load_full();
        let mut new_map = (*old).clone();
        new_map.insert(key, Arc::new(value));
        self.inner.store(Arc::new(new_map));
        drop(guard);

        let count = self.update_count.fetch_add(1, Ordering::Relaxed) + 1;
        info!(
            target: "ffs::mvcc::rcu",
            update_count = count,
            "rcu_map_update"
        );
        if count % self.churn_threshold == 0 {
            warn!(
                target: "ffs::mvcc::rcu",
                update_count = count,
                churn_threshold = self.churn_threshold,
                "rcu_map_high_churn"
            );
        }
    }

    /// Remove a key (copy-on-write).
    ///
    /// Returns `true` if the key was present and removed.
    pub fn remove(&self, key: &K) -> bool {
        let guard = self.write_lock.lock();
        let old = self.inner.load_full();
        if !old.contains_key(key) {
            return false;
        }
        let mut new_map = (*old).clone();
        new_map.remove(key);
        self.inner.store(Arc::new(new_map));
        drop(guard);

        self.update_count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Number of entries in the current snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.load().len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.load().is_empty()
    }

    /// Total number of updates since creation.
    #[must_use]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Replace the entire map atomically.
    pub fn replace(&self, new_map: BTreeMap<K, Arc<V>>) {
        let _guard = self.write_lock.lock();
        self.inner.store(Arc::new(new_map));
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Clear all entries (publish an empty map).
    pub fn clear(&self) {
        let _guard = self.write_lock.lock();
        self.inner.store(Arc::new(BTreeMap::new()));
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }
}

impl<K: Clone + Ord + Hash + fmt::Debug, V: Clone + fmt::Debug> Default for RcuMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── AtomicWatermark ────────────────────────────────────────────────────────

/// Lock-free watermark for snapshot GC.
///
/// Stores a `CommitSeq` (u64) that can be read atomically without any lock.
/// Writers update it when the set of active snapshots changes.
///
/// The sentinel value `u64::MAX` represents "no active snapshots" (empty).
#[derive(Debug)]
pub struct AtomicWatermark {
    value: AtomicU64,
}

const WATERMARK_EMPTY: u64 = u64::MAX;

impl AtomicWatermark {
    /// Create with no active snapshots.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(WATERMARK_EMPTY),
        }
    }

    /// Create with an initial watermark value.
    #[must_use]
    pub fn with_value(commit_seq: u64) -> Self {
        Self {
            value: AtomicU64::new(commit_seq),
        }
    }

    /// Load the current watermark.
    ///
    /// Returns `None` if no active snapshots exist.
    /// Completely lock-free — just an atomic load.
    #[inline]
    #[must_use]
    pub fn load(&self) -> Option<u64> {
        let v = self.value.load(Ordering::Acquire);
        if v == WATERMARK_EMPTY { None } else { Some(v) }
    }

    /// Store a new watermark value.
    #[inline]
    pub fn store(&self, commit_seq: u64) {
        self.value.store(commit_seq, Ordering::Release);
    }

    /// Clear the watermark (no active snapshots).
    #[inline]
    pub fn clear(&self) {
        self.value.store(WATERMARK_EMPTY, Ordering::Release);
    }

    /// Load the raw u64 (including sentinel).
    #[inline]
    #[must_use]
    pub fn load_raw(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }
}

impl Default for AtomicWatermark {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn rcu_cell_basic_read_write() {
        let cell = RcuCell::new(42_u64);
        assert_eq!(**cell.load(), 42);

        cell.update(100);
        assert_eq!(**cell.load(), 100);
        assert_eq!(cell.update_count(), 1);
    }

    #[test]
    fn rcu_cell_arc_roundtrip() {
        let cell = RcuCell::from_arc(Arc::new("hello".to_string()));
        assert_eq!(&***cell.load(), "hello");

        cell.update("world".to_string());
        assert_eq!(&***cell.load(), "world");
    }

    #[test]
    fn rcu_cell_swap_returns_old() {
        let cell = RcuCell::new(1_u32);
        let old = cell.swap(2);
        assert_eq!(*old, 1);
        assert_eq!(**cell.load(), 2);
    }

    #[test]
    fn rcu_cell_concurrent_readers_no_block() {
        let cell = Arc::new(RcuCell::new(0_u64));
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cell = Arc::clone(&cell);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut sum = 0_u64;
                    for _ in 0..10_000 {
                        sum = sum.wrapping_add(**cell.load());
                    }
                    sum
                })
            })
            .collect();

        for h in handles {
            let _ = h.join().unwrap();
        }
    }

    #[test]
    fn rcu_cell_readers_see_consistent_value() {
        let cell = Arc::new(RcuCell::new(0_u64));

        // Writer thread
        let writer_cell = Arc::clone(&cell);
        let writer = thread::spawn(move || {
            for i in 1..=1000 {
                writer_cell.update(i);
            }
        });

        // Reader threads
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let cell = Arc::clone(&cell);
                thread::spawn(move || {
                    let mut prev = 0_u64;
                    for _ in 0..10_000 {
                        let val = **cell.load();
                        // Values must be monotonically non-decreasing
                        // (single writer, sequential updates).
                        assert!(val >= prev, "non-monotonic: {val} < {prev}");
                        prev = val;
                    }
                })
            })
            .collect();

        writer.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn rcu_map_basic_operations() {
        let map: RcuMap<u64, String> = RcuMap::new();
        assert!(map.is_empty());

        map.insert(1, "one".to_string());
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&1).map(|v| v.as_str().to_owned()),
            Some("one".to_owned())
        );

        map.insert(2, "two".to_string());
        assert_eq!(map.len(), 2);

        assert!(map.remove(&1));
        assert_eq!(map.len(), 1);
        assert!(map.get(&1).is_none());

        assert!(!map.remove(&99));
    }

    #[test]
    fn rcu_map_concurrent_read_write() {
        let map = Arc::new(RcuMap::<u64, u64>::new());
        let barrier = Arc::new(Barrier::new(5));

        // Writer
        let writer_map = Arc::clone(&map);
        let writer_barrier = Arc::clone(&barrier);
        let writer = thread::spawn(move || {
            writer_barrier.wait();
            for i in 0..500 {
                writer_map.insert(i, i * 10);
            }
        });

        // Readers
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let map = Arc::clone(&map);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut reads = 0_u64;
                    for _ in 0..5_000 {
                        let snap = map.load();
                        // Snapshot must be internally consistent
                        for (k, v) in snap.iter() {
                            assert_eq!(**v, *k * 10, "inconsistent: k={k}, v={v}");
                        }
                        reads += 1;
                    }
                    reads
                })
            })
            .collect();

        writer.join().unwrap();
        for r in readers {
            let reads = r.join().unwrap();
            assert!(reads > 0);
        }
    }

    #[test]
    fn rcu_map_snapshot_isolation() {
        let map: RcuMap<u64, String> = RcuMap::new();
        map.insert(1, "original".to_string());

        // Take a snapshot
        let snapshot = map.load_arc();
        assert_eq!(snapshot.get(&1).unwrap().as_str(), "original");

        // Update the map
        map.insert(1, "updated".to_string());

        // Old snapshot still sees original value
        assert_eq!(snapshot.get(&1).unwrap().as_str(), "original");

        // New load sees updated value
        assert_eq!(map.get(&1).unwrap().as_str(), "updated");
    }

    #[test]
    fn rcu_map_replace_and_clear() {
        let map: RcuMap<u64, u64> = RcuMap::new();
        map.insert(1, 10);
        map.insert(2, 20);

        let mut new = BTreeMap::new();
        new.insert(3, Arc::new(30));
        map.replace(new);

        assert_eq!(map.len(), 1);
        assert!(map.get(&1).is_none());
        assert_eq!(*map.get(&3).unwrap(), 30);

        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn atomic_watermark_basic() {
        let wm = AtomicWatermark::new();
        assert_eq!(wm.load(), None);

        wm.store(42);
        assert_eq!(wm.load(), Some(42));

        wm.clear();
        assert_eq!(wm.load(), None);
    }

    #[test]
    fn atomic_watermark_concurrent_reads() {
        let wm = Arc::new(AtomicWatermark::with_value(100));
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let wm = Arc::clone(&wm);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    let mut reads = 0_u64;
                    for _ in 0..100_000 {
                        if wm.load().is_some() {
                            reads += 1;
                        }
                    }
                    reads
                })
            })
            .collect();

        for h in handles {
            let reads = h.join().unwrap();
            assert_eq!(reads, 100_000);
        }
    }

    #[test]
    fn rcu_cell_update_arc_publishes_value() {
        let cell = RcuCell::new(10_u64);
        cell.update_arc(Arc::new(42));
        assert_eq!(**cell.load(), 42);
        assert_eq!(cell.update_count(), 1);
    }

    #[test]
    fn rcu_cell_load_arc_returns_independent_arc() {
        let cell = RcuCell::new(99_u32);
        let arc1 = cell.load_arc();
        let arc2 = cell.load_arc();
        assert_eq!(*arc1, 99);
        assert_eq!(*arc2, 99);
        // update doesn't affect already-loaded arcs
        cell.update(200);
        assert_eq!(*arc1, 99);
        assert_eq!(**cell.load(), 200);
    }

    #[test]
    fn rcu_map_with_churn_threshold_sets_threshold() {
        let map: RcuMap<u64, u64> = RcuMap::with_churn_threshold(5);
        // Insert 5 entries — should hit the churn threshold
        for i in 0..5 {
            map.insert(i, i * 10);
        }
        assert_eq!(map.update_count(), 5);
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn rcu_map_default_is_empty() {
        let map: RcuMap<u64, u64> = RcuMap::default();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.update_count(), 0);
    }

    #[test]
    fn atomic_watermark_with_value_constructor() {
        let wm = AtomicWatermark::with_value(42);
        assert_eq!(wm.load(), Some(42));
        assert_eq!(wm.load_raw(), 42);
    }

    #[test]
    fn atomic_watermark_max_sentinel_reads_as_none() {
        let wm = AtomicWatermark::new();
        assert_eq!(wm.load(), None);
        assert_eq!(wm.load_raw(), u64::MAX);
    }

    #[test]
    fn rcu_cell_debug_format() {
        let cell = RcuCell::new(42_u64);
        let debug_str = format!("{cell:?}");
        assert!(debug_str.contains("RcuCell"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn rcu_map_debug_format() {
        let map: RcuMap<u64, u64> = RcuMap::new();
        map.insert(1, 10);
        let debug_str = format!("{map:?}");
        assert!(debug_str.contains("RcuMap"));
        assert!(debug_str.contains("entry_count"));
    }

    // ── Property-based tests (proptest) ────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// RcuCell: update then load always returns the latest value.
        #[test]
        fn proptest_rcu_cell_update_load_consistency(
            values in proptest::collection::vec(any::<u64>(), 1..16),
        ) {
            let cell = RcuCell::new(0_u64);
            for &v in &values {
                cell.update(v);
                prop_assert_eq!(**cell.load(), v);
            }
            prop_assert_eq!(cell.update_count(), values.len() as u64);
        }

        /// RcuCell: swap returns the previous value.
        #[test]
        fn proptest_rcu_cell_swap_returns_previous(
            initial in any::<u32>(),
            new_val in any::<u32>(),
        ) {
            let cell = RcuCell::new(initial);
            let old = cell.swap(new_val);
            prop_assert_eq!(*old, initial);
            prop_assert_eq!(**cell.load(), new_val);
        }

        /// RcuCell: from_arc roundtrip preserves value.
        #[test]
        fn proptest_rcu_cell_from_arc_roundtrip(value in any::<u64>()) {
            let cell = RcuCell::from_arc(Arc::new(value));
            prop_assert_eq!(**cell.load(), value);
            let arc = cell.load_arc();
            prop_assert_eq!(*arc, value);
        }

        /// RcuMap: insert/get roundtrip for arbitrary keys and values.
        #[test]
        fn proptest_rcu_map_insert_get_roundtrip(
            entries in proptest::collection::vec((1_u64..256, any::<u64>()), 1..16),
        ) {
            let map: RcuMap<u64, u64> = RcuMap::new();
            let mut expected = std::collections::BTreeMap::new();
            for &(k, v) in &entries {
                map.insert(k, v);
                expected.insert(k, v);
            }
            for (&k, &v) in &expected {
                let got = map.get(&k).expect("key must exist");
                prop_assert_eq!(*got, v, "mismatch for key {}", k);
            }
            prop_assert_eq!(map.len(), expected.len());
        }

        /// RcuMap: remove returns true for existing keys, false for missing.
        #[test]
        fn proptest_rcu_map_remove_semantics(
            key in 1_u64..100,
            value in any::<u64>(),
        ) {
            let map: RcuMap<u64, u64> = RcuMap::new();
            prop_assert!(!map.remove(&key));  // not yet inserted
            map.insert(key, value);
            prop_assert!(map.remove(&key));   // now present
            prop_assert!(map.get(&key).is_none());
        }

        /// RcuMap: snapshot isolation — old load_arc sees original state.
        #[test]
        fn proptest_rcu_map_snapshot_isolation(
            initial in any::<u64>(),
            updated in any::<u64>(),
        ) {
            let map: RcuMap<u64, u64> = RcuMap::new();
            map.insert(1, initial);
            let snapshot = map.load_arc();
            map.insert(1, updated);
            // Old snapshot sees original
            prop_assert_eq!(**snapshot.get(&1).unwrap(), initial);
            // New read sees updated
            prop_assert_eq!(*map.get(&1).unwrap(), updated);
        }

        /// AtomicWatermark: store/load roundtrip.
        #[test]
        fn proptest_atomic_watermark_store_load(value in any::<u64>()) {
            // Exclude u64::MAX since it's the sentinel for "empty"
            if value != u64::MAX {
                let wm = AtomicWatermark::new();
                wm.store(value);
                prop_assert_eq!(wm.load(), Some(value));
                prop_assert_eq!(wm.load_raw(), value);
            }
        }

        /// AtomicWatermark: clear sets load to None.
        #[test]
        fn proptest_atomic_watermark_clear(value in 0_u64..u64::MAX) {
            let wm = AtomicWatermark::with_value(value);
            prop_assert_eq!(wm.load(), Some(value));
            wm.clear();
            prop_assert_eq!(wm.load(), None);
        }
    }
}
