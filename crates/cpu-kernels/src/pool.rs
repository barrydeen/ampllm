use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;

#[repr(C)]
struct TaskEnvelope {
    total: usize,
    fn_ptr: *const (),
    trampoline: unsafe fn(*const (), usize, usize),
}

unsafe impl Send for TaskEnvelope {}
unsafe impl Sync for TaskEnvelope {}

struct SharedState {
    epoch: AtomicU64,
    done: AtomicUsize,
    shutdown: AtomicBool,
    parked_mask: AtomicU64,
    task: AtomicPtr<TaskEnvelope>,
}

/// Physical-core pinned spin-barrier thread pool for low-latency CPU cold MoE decode.
pub struct CpuThreadPool {
    shared: Arc<SharedState>,
    workers: Vec<JoinHandle<()>>,
    worker_threads: Vec<Thread>,
    n_threads: usize,
    physical_cores: Vec<usize>,
}

/// Discover one logical CPU ID per physical core.
/// When `exclude_cpu0` is true, CPU 0 is left free for OS/CUDA dispatch.
pub fn discover_physical_cores_with_options(
    max_cores: Option<usize>,
    exclude_cpu0: bool,
) -> Vec<usize> {
    let mut seen_groups: BTreeSet<usize> = BTreeSet::new();
    let mut cores = Vec::new();

    if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
        let mut cpu_ids: Vec<usize> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let s = name.to_str()?;
                s.strip_prefix("cpu")?.parse::<usize>().ok()
            })
            .collect();
        cpu_ids.sort_unstable();

        for cpu_id in cpu_ids {
            let sib_path = format!(
                "/sys/devices/system/cpu/cpu{cpu_id}/topology/thread_siblings_list"
            );
            let first_sib = if let Ok(txt) = std::fs::read_to_string(Path::new(&sib_path)) {
                parse_first_cpu_in_list(txt.trim()).unwrap_or(cpu_id)
            } else {
                cpu_id
            };
            if seen_groups.insert(first_sib) {
                cores.push(first_sib);
            }
        }
        if exclude_cpu0 && cores.len() > 8 {
            cores.retain(|&c| c != 0);
        }
    }

    if cores.is_empty() {
        let avail = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let fallback = (avail / 2).max(1);
        let skip0 = usize::from(exclude_cpu0 && fallback > 8);
        for c in skip0..fallback {
            cores.push(c);
        }
        if cores.is_empty() {
            cores.push(0);
        }
    }

    if let Some(limit) = max_cores {
        let cap = limit.max(1);
        if cores.len() > cap {
            cores.truncate(cap);
        }
    }
    cores
}

pub fn discover_physical_cores(max_cores: Option<usize>) -> Vec<usize> {
    discover_physical_cores_with_options(max_cores, true)
}

/// Discover physical cores partitioned into two L3-cache (CCD) groups when available.
pub fn discover_ccd_split_cores() -> (Vec<usize>, Vec<usize>) {
    let cores = discover_physical_cores_with_options(None, false);
    let mut by_l3: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &c in &cores {
        let p = format!("/sys/devices/system/cpu/cpu{c}/cache/index3/id");
        let l3_id = std::fs::read_to_string(Path::new(&p))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        by_l3.entry(l3_id).or_default().push(c);
    }
    if by_l3.len() >= 2 {
        let mut it = by_l3.into_values();
        let g0 = it.next().unwrap();
        let g1 = it.next().unwrap();
        return (g0, g1);
    }
    let mid = (cores.len() / 2).max(1);
    let g0 = cores[..mid].to_vec();
    let g1 = if mid < cores.len() {
        cores[mid..].to_vec()
    } else {
        cores.clone()
    };
    (g0, g1)
}

/// Discover all logical CPU IDs (including SMT siblings) partitioned by L3-cache (CCD)
/// group, excluding CPU 0 from its CCD when multiple CPUs exist there.
pub fn discover_ccd_split_cores_smt() -> (Vec<usize>, Vec<usize>) {
    let mut cpu_ids: Vec<usize> = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
        cpu_ids = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let s = name.to_str()?;
                s.strip_prefix("cpu")?.parse::<usize>().ok()
            })
            .collect();
        cpu_ids.sort_unstable();
    }
    if cpu_ids.is_empty() {
        return discover_ccd_split_cores();
    }
    let mut by_l3: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &c in &cpu_ids {
        let p = format!("/sys/devices/system/cpu/cpu{c}/cache/index3/id");
        let l3_id = std::fs::read_to_string(Path::new(&p))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        by_l3.entry(l3_id).or_default().push(c);
    }
    if by_l3.len() >= 2 {
        let mut it = by_l3.into_values();
        let mut g0 = it.next().unwrap();
        let mut g1 = it.next().unwrap();
        if g0.len() > 2 {
            g0.retain(|&c| c != 0);
        }
        if g1.len() > 2 {
            g1.retain(|&c| c != 0);
        }
        return (g0, g1);
    }
    if cpu_ids.len() > 2 {
        cpu_ids.retain(|&c| c != 0);
    }
    let mid = (cpu_ids.len() / 2).max(1);
    let g0 = cpu_ids[..mid].to_vec();
    let g1 = if mid < cpu_ids.len() {
        cpu_ids[mid..].to_vec()
    } else {
        g0.clone()
    };
    (g0, g1)
}

/// Discover physical cores interleaved across both CCDs (excluding CPU 0 for OS/CUDA dispatch)
/// so each pool spans both GMI3 read links (~55 GB/s vs ~30 GB/s single-CCD).
pub fn discover_cross_ccd_interleaved_cores() -> (Vec<usize>, Vec<usize>) {
    let (ccd0, ccd1) = discover_ccd_split_cores();
    if ccd0.len() >= 4 && ccd1.len() >= 4 {
        let ccd0_no0: Vec<usize> = ccd0.into_iter().filter(|&c| c != 0).collect();
        let mid0 = ccd0_no0.len() / 2;
        let mid1 = ccd1.len() / 2;
        let mut p0 = Vec::with_capacity(mid0 + mid1);
        p0.extend_from_slice(&ccd0_no0[..mid0]);
        p0.extend_from_slice(&ccd1[..mid1]);
        let mut p1 = Vec::with_capacity((ccd0_no0.len() - mid0) + (ccd1.len() - mid1));
        p1.extend_from_slice(&ccd0_no0[mid0..]);
        p1.extend_from_slice(&ccd1[mid1..]);
        return (p0, p1);
    }
    let mut cores = discover_physical_cores_with_options(None, false);
    if cores.len() > 2 && cores[0] == 0 {
        cores.remove(0);
    }
    let mid = (cores.len() / 2).max(1);
    let g0 = cores[..mid].to_vec();
    let g1 = if mid < cores.len() {
        cores[mid..].to_vec()
    } else {
        cores
    };
    (g0, g1)
}

fn parse_first_cpu_in_list(s: &str) -> Option<usize> {
    let first_part = s.split(',').next()?.trim();
    let first_num = first_part.split('-').next()?.trim();
    first_num.parse::<usize>().ok()
}

pub fn pin_current_thread(cpu_id: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        if cpu_id < libc::CPU_SETSIZE as usize {
            libc::CPU_SET(cpu_id, &mut set);
            let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

unsafe fn call_closure<F: Fn(usize, usize) + Sync>(ptr: *const (), start: usize, end: usize) {
    let f = &*(ptr as *const F);
    f(start, end);
}

impl CpuThreadPool {
    pub fn new(num_threads: Option<usize>) -> Self {
        Self::with_options(num_threads, true)
    }

    pub fn new_all_physical(num_threads: Option<usize>) -> Self {
        Self::with_options(num_threads, false)
    }

    pub fn with_options(num_threads: Option<usize>, exclude_cpu0: bool) -> Self {
        let physical_cores = discover_physical_cores_with_options(num_threads, exclude_cpu0);
        let n_threads = match num_threads {
            Some(n) => n.max(1).min(physical_cores.len().max(1)),
            None => physical_cores.len().max(1),
        };
        let used_cores = physical_cores[..n_threads.min(physical_cores.len())].to_vec();
        Self::from_cores(used_cores)
    }

    pub fn from_cores(cores: Vec<usize>) -> Self {
        let used_cores = if cores.is_empty() { vec![0] } else { cores };
        let n_threads = used_cores.len();

        let shared = Arc::new(SharedState {
            epoch: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            parked_mask: AtomicU64::new(0),
            task: AtomicPtr::new(std::ptr::null_mut()),
        });

        let mut workers = Vec::with_capacity(n_threads.saturating_sub(1));
        let mut worker_threads = Vec::with_capacity(n_threads.saturating_sub(1));

        for tid in 1..n_threads {
            let shared_c = Arc::clone(&shared);
            let core_id = used_cores.get(tid).copied().unwrap_or(tid);
            let handle = thread::Builder::new()
                .name(format!("ampllm-cpu-{tid}"))
                .spawn(move || {
                    pin_current_thread(core_id);
                    let mut seen_epoch = 0u64;
                    let bit = if tid < 64 { 1u64 << tid } else { 0u64 };

                    loop {
                        let mut spins = 0u32;
                        loop {
                            let cur = shared_c.epoch.load(Ordering::Acquire);
                            if cur != seen_epoch {
                                seen_epoch = cur;
                                break;
                            }
                            if shared_c.shutdown.load(Ordering::Acquire) {
                                return;
                            }
                            if spins < 200_000 {
                                std::hint::spin_loop();
                                spins += 1;
                            } else if spins < 202_000 {
                                thread::yield_now();
                                spins += 1;
                            } else {
                                if bit != 0 {
                                    shared_c.parked_mask.fetch_or(bit, Ordering::Release);
                                }
                                if shared_c.epoch.load(Ordering::Acquire) == seen_epoch
                                    && !shared_c.shutdown.load(Ordering::Acquire)
                                {
                                    thread::park_timeout(Duration::from_micros(250));
                                }
                                if bit != 0 {
                                    shared_c.parked_mask.fetch_and(!bit, Ordering::Release);
                                }
                            }
                        }

                        let task_ptr = shared_c.task.load(Ordering::Acquire);
                        if !task_ptr.is_null() {
                            let env = unsafe { &*task_ptr };
                            let (s, e) = slice_bounds(env.total, tid, n_threads);
                            if s < e {
                                unsafe { (env.trampoline)(env.fn_ptr, s, e) };
                            }
                        }
                        shared_c.done.fetch_add(1, Ordering::AcqRel);
                    }
                })
                .expect("failed to spawn cpu worker thread");
            worker_threads.push(handle.thread().clone());
            workers.push(handle);
        }

        Self {
            shared,
            workers,
            worker_threads,
            n_threads,
            physical_cores: used_cores,
        }
    }

    pub fn num_threads(&self) -> usize {
        self.n_threads
    }

    pub fn physical_cores(&self) -> &[usize] {
        &self.physical_cores
    }

    pub fn parallel_for<F>(&self, total: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if total == 0 {
            return;
        }
        if self.n_threads == 1 || total <= 1 {
            f(0, total);
            return;
        }

        let mut env = TaskEnvelope {
            total,
            fn_ptr: (&f as *const F) as *const (),
            trampoline: call_closure::<F>,
        };

        self.shared.done.store(0, Ordering::Release);
        self.shared
            .task
            .store(&mut env as *mut TaskEnvelope, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::AcqRel);

        let parked = self.shared.parked_mask.load(Ordering::Acquire);
        if parked != 0 {
            for (idx, th) in self.worker_threads.iter().enumerate() {
                let tid = idx + 1;
                if tid >= 64 || (parked & (1u64 << tid)) != 0 {
                    th.unpark();
                }
            }
        }

        let (s0, e0) = slice_bounds(total, 0, self.n_threads);
        if s0 < e0 {
            f(s0, e0);
        }

        let expected = self.n_threads - 1;
        while self.shared.done.load(Ordering::Acquire) < expected {
            std::hint::spin_loop();
        }
        self.shared.task.store(std::ptr::null_mut(), Ordering::Release);
    }
}

#[inline]
fn slice_bounds(total: usize, tid: usize, n_threads: usize) -> (usize, usize) {
    let base = total / n_threads;
    let rem = total % n_threads;
    let start = tid * base + tid.min(rem);
    let len = base + if tid < rem { 1 } else { 0 };
    (start, start + len)
}

impl Drop for CpuThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.epoch.fetch_add(1, Ordering::AcqRel);
        for th in &self.worker_threads {
            th.unpark();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}
