use anyhow::{bail, Result};
use libc::{c_char, c_int, c_uint, c_void};
use std::ffi::CStr;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

pub const CUDA_MEMCPY_HOST_TO_HOST: c_int = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

pub const CUDA_HOST_ALLOC_PORTABLE: c_uint = 0x01;
pub const CUDA_HOST_ALLOC_MAPPED: c_uint = 0x02;
pub const CUDA_HOST_REGISTER_PORTABLE: c_uint = 0x01;
pub const CUDA_HOST_REGISTER_MAPPED: c_uint = 0x02;

pub const CUDA_STREAM_NON_BLOCKING: c_uint = 0x01;
pub const CUDA_EVENT_DISABLE_TIMING: c_uint = 0x02;
pub const CUDA_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CudaStreamHandle(pub *mut c_void);
unsafe impl Send for CudaStreamHandle {}
unsafe impl Sync for CudaStreamHandle {}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CudaEventHandle(pub *mut c_void);
unsafe impl Send for CudaEventHandle {}
unsafe impl Sync for CudaEventHandle {}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CudaGraphHandle(pub *mut c_void);
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CudaGraphExecHandle(pub *mut c_void);
unsafe impl Send for CudaGraphExecHandle {}
unsafe impl Sync for CudaGraphExecHandle {}

extern "C" {
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
    fn cudaDeviceCanAccessPeer(can_access: *mut c_int, dev: c_int, peer_dev: c_int) -> c_int;
    fn cudaDeviceEnablePeerAccess(peer_dev: c_int, flags: c_uint) -> c_int;

    fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    fn cudaMemset(dev_ptr: *mut c_void, value: c_int, count: usize) -> c_int;
    fn cudaMemsetAsync(dev_ptr: *mut c_void, value: c_int, count: usize, stream: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaMemcpyPeerAsync(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;

    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaHostGetDevicePointer(p_device: *mut *mut c_void, p_host: *mut c_void, flags: c_uint) -> c_int;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: c_uint) -> c_int;
    fn cudaHostUnregister(ptr: *mut c_void) -> c_int;

    fn cudaStreamCreateWithFlags(p_stream: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaStreamCreateWithPriority(p_stream: *mut *mut c_void, flags: c_uint, priority: c_int) -> c_int;
    fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: c_uint) -> c_int;

    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: c_uint) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    fn cudaEventQuery(event: *mut c_void) -> c_int;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> c_int;

    fn cudaStreamBeginCapture(stream: *mut c_void, mode: c_int) -> c_int;
    fn cudaStreamEndCapture(stream: *mut c_void, p_graph: *mut *mut c_void) -> c_int;
    fn cudaGraphInstantiateWithFlags(
        p_graph_exec: *mut *mut c_void,
        graph: *mut c_void,
        flags: u64,
    ) -> c_int;
    fn cudaGraphLaunch(graph_exec: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaGraphDestroy(graph: *mut c_void) -> c_int;
    fn cudaGraphExecDestroy(graph_exec: *mut c_void) -> c_int;

    fn cudaGetErrorString(err: c_int) -> *const c_char;
    fn cudaGetLastError() -> c_int;
}

#[inline]
pub fn cuda_check(code: c_int, op: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(cudaGetErrorString(code)) }
        .to_string_lossy()
        .into_owned();
    bail!("CUDA error during {op} (code {code}): {msg}")
}

pub fn device_count() -> Result<i32> {
    let mut n = 0;
    cuda_check(unsafe { cudaGetDeviceCount(&mut n) }, "cudaGetDeviceCount")?;
    Ok(n)
}

pub fn set_device(device: i32) -> Result<()> {
    cuda_check(unsafe { cudaSetDevice(device) }, "cudaSetDevice")
}

pub fn current_device() -> Result<i32> {
    let mut d = 0;
    cuda_check(unsafe { cudaGetDevice(&mut d) }, "cudaGetDevice")?;
    Ok(d)
}

pub fn device_synchronize() -> Result<()> {
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

pub fn mem_info(device: i32) -> Result<(usize, usize)> {
    set_device(device)?;
    let (mut free, mut total) = (0usize, 0usize);
    cuda_check(unsafe { cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok((free, total))
}

pub fn can_access_peer(dev: i32, peer: i32) -> Result<bool> {
    let mut ok = 0;
    cuda_check(
        unsafe { cudaDeviceCanAccessPeer(&mut ok, dev, peer) },
        "cudaDeviceCanAccessPeer",
    )?;
    Ok(ok != 0)
}

pub fn enable_peer_access(dev: i32, peer: i32) -> Result<bool> {
    if !can_access_peer(dev, peer)? {
        return Ok(false);
    }
    set_device(dev)?;
    let code = unsafe { cudaDeviceEnablePeerAccess(peer, 0) };
    // 704 == cudaErrorPeerAccessAlreadyEnabled
    if code == 0 || code == 704 {
        Ok(true)
    } else {
        cuda_check(code, "cudaDeviceEnablePeerAccess")?;
        Ok(true)
    }
}

pub fn stream_priority_range(device: i32) -> Result<(i32, i32)> {
    set_device(device)?;
    let (mut least, mut greatest) = (0, 0);
    cuda_check(
        unsafe { cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest) },
        "cudaDeviceGetStreamPriorityRange",
    )?;
    Ok((least, greatest))
}

pub struct CudaStream {
    pub device: i32,
    pub raw: CudaStreamHandle,
}

impl CudaStream {
    pub fn new(device: i32) -> Result<Self> {
        set_device(device)?;
        let mut raw = ptr::null_mut();
        cuda_check(
            unsafe { cudaStreamCreateWithFlags(&mut raw, CUDA_STREAM_NON_BLOCKING) },
            "cudaStreamCreateWithFlags",
        )?;
        Ok(Self {
            device,
            raw: CudaStreamHandle(raw),
        })
    }

    pub fn with_priority(device: i32, priority: i32) -> Result<Self> {
        set_device(device)?;
        let mut raw = ptr::null_mut();
        cuda_check(
            unsafe { cudaStreamCreateWithPriority(&mut raw, CUDA_STREAM_NON_BLOCKING, priority) },
            "cudaStreamCreateWithPriority",
        )?;
        Ok(Self {
            device,
            raw: CudaStreamHandle(raw),
        })
    }

    pub fn sync(&self) -> Result<()> {
        cuda_check(unsafe { cudaStreamSynchronize(self.raw.0) }, "cudaStreamSynchronize")
    }

    pub fn wait_event(&self, event: &CudaEvent) -> Result<()> {
        cuda_check(
            unsafe { cudaStreamWaitEvent(self.raw.0, event.raw.0, 0) },
            "cudaStreamWaitEvent",
        )
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.0.is_null() {
            let _ = unsafe { cudaStreamDestroy(self.raw.0) };
        }
    }
}

pub struct CudaEvent {
    pub raw: CudaEventHandle,
}

impl CudaEvent {
    pub fn new(device: i32) -> Result<Self> {
        set_device(device)?;
        let mut raw = ptr::null_mut();
        cuda_check(
            unsafe { cudaEventCreateWithFlags(&mut raw, CUDA_EVENT_DISABLE_TIMING) },
            "cudaEventCreateWithFlags",
        )?;
        Ok(Self {
            raw: CudaEventHandle(raw),
        })
    }

    pub fn new_timed(device: i32) -> Result<Self> {
        set_device(device)?;
        let mut raw = ptr::null_mut();
        cuda_check(
            unsafe { cudaEventCreateWithFlags(&mut raw, 0) },
            "cudaEventCreateWithFlags(timed)",
        )?;
        Ok(Self {
            raw: CudaEventHandle(raw),
        })
    }

    pub fn elapsed_ms(&self, end: &CudaEvent) -> Result<f32> {
        let mut ms = 0.0f32;
        cuda_check(
            unsafe { cudaEventElapsedTime(&mut ms, self.raw.0, end.raw.0) },
            "cudaEventElapsedTime",
        )?;
        Ok(ms)
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        cuda_check(
            unsafe { cudaEventRecord(self.raw.0, stream.raw.0) },
            "cudaEventRecord",
        )
    }

    pub fn sync(&self) -> Result<()> {
        cuda_check(unsafe { cudaEventSynchronize(self.raw.0) }, "cudaEventSynchronize")
    }

    pub fn is_ready(&self) -> Result<bool> {
        let code = unsafe { cudaEventQuery(self.raw.0) };
        if code == 0 {
            Ok(true)
        } else if code == 600 {
            // cudaErrorNotReady
            Ok(false)
        } else {
            cuda_check(code, "cudaEventQuery")?;
            Ok(false)
        }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.0.is_null() {
            let _ = unsafe { cudaEventDestroy(self.raw.0) };
        }
    }
}

pub struct DeviceBuffer {
    pub device: i32,
    pub ptr: *mut c_void,
    pub bytes: usize,
}
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn alloc(device: i32, bytes: usize) -> Result<Self> {
        set_device(device)?;
        let alloc_bytes = bytes.max(16);
        let mut ptr = ptr::null_mut();
        let code = unsafe { cudaMalloc(&mut ptr, alloc_bytes) };
        if code != 0 {
            let _ = unsafe { cudaGetLastError() };
            cuda_check(code, "cudaMalloc")?;
        }
        Ok(Self {
            device,
            ptr,
            bytes,
        })
    }

    pub fn alloc_zeroed(device: i32, bytes: usize) -> Result<Self> {
        let buf = Self::alloc(device, bytes)?;
        buf.zero()?;
        Ok(buf)
    }

    pub fn zero(&self) -> Result<()> {
        if self.bytes > 0 {
            set_device(self.device)?;
            cuda_check(unsafe { cudaMemset(self.ptr, 0, self.bytes) }, "cudaMemset")?;
            // cudaMemset is asynchronous on legacy stream 0; sync so non-blocking streams observe zeroed memory.
            cuda_check(
                unsafe { cudaStreamSynchronize(ptr::null_mut()) },
                "cudaStreamSynchronize(default)",
            )?;
        }
        Ok(())
    }

    pub fn zero_async(&self, stream: &CudaStream) -> Result<()> {
        if self.bytes > 0 {
            cuda_check(
                unsafe { cudaMemsetAsync(self.ptr, 0, self.bytes, stream.raw.0) },
                "cudaMemsetAsync",
            )?;
        }
        Ok(())
    }

    pub fn copy_from_host(&self, src: &[u8]) -> Result<()> {
        if src.len() > self.bytes {
            bail!("copy_from_host overflow: {} > {}", src.len(), self.bytes);
        }
        if !src.is_empty() {
            set_device(self.device)?;
            cuda_check(
                unsafe {
                    cudaMemcpy(
                        self.ptr,
                        src.as_ptr().cast(),
                        src.len(),
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "cudaMemcpy H2D",
            )?;
            // Wait for legacy stream 0 so non-blocking streams immediately observe uploaded bytes.
            cuda_check(
                unsafe { cudaStreamSynchronize(ptr::null_mut()) },
                "cudaStreamSynchronize(default)",
            )?;
        }
        Ok(())
    }

    pub fn copy_from_host_async(&self, src: &[u8], stream: &CudaStream) -> Result<()> {
        if src.len() > self.bytes {
            bail!("copy_from_host_async overflow: {} > {}", src.len(), self.bytes);
        }
        if !src.is_empty() {
            cuda_check(
                unsafe {
                    cudaMemcpyAsync(
                        self.ptr,
                        src.as_ptr().cast(),
                        src.len(),
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        stream.raw.0,
                    )
                },
                "cudaMemcpyAsync H2D",
            )?;
        }
        Ok(())
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() > self.bytes {
            bail!("copy_to_host overflow: {} > {}", dst.len(), self.bytes);
        }
        if !dst.is_empty() {
            set_device(self.device)?;
            cuda_check(
                unsafe {
                    cudaMemcpy(
                        dst.as_mut_ptr().cast(),
                        self.ptr,
                        dst.len(),
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                    )
                },
                "cudaMemcpy D2H",
            )?;
        }
        Ok(())
    }

    pub fn copy_to_host_async(&self, dst: &mut [u8], stream: &CudaStream) -> Result<()> {
        if dst.len() > self.bytes {
            bail!("copy_to_host_async overflow: {} > {}", dst.len(), self.bytes);
        }
        if !dst.is_empty() {
            cuda_check(
                unsafe {
                    cudaMemcpyAsync(
                        dst.as_mut_ptr().cast(),
                        self.ptr,
                        dst.len(),
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        stream.raw.0,
                    )
                },
                "cudaMemcpyAsync D2H",
            )?;
        }
        Ok(())
    }

    pub fn copy_from_device_async(&self, src: &DeviceBuffer, bytes: usize, stream: &CudaStream) -> Result<()> {
        if bytes > self.bytes || bytes > src.bytes {
            bail!("copy_from_device_async size mismatch");
        }
        if bytes > 0 {
            if self.device == src.device {
                cuda_check(
                    unsafe {
                        cudaMemcpyAsync(
                            self.ptr,
                            src.ptr,
                            bytes,
                            CUDA_MEMCPY_DEVICE_TO_DEVICE,
                            stream.raw.0,
                        )
                    },
                    "cudaMemcpyAsync D2D",
                )?;
            } else {
                cuda_check(
                    unsafe {
                        cudaMemcpyPeerAsync(
                            self.ptr,
                            self.device,
                            src.ptr,
                            src.device,
                            bytes,
                            stream.raw.0,
                        )
                    },
                    "cudaMemcpyPeerAsync",
                )?;
            }
        }
        Ok(())
    }

    pub fn as_u64(&self) -> u64 {
        self.ptr as u64
    }
}

pub unsafe fn copy_h2d_raw_async(dst_dev: *mut c_void, src_host: &[u8], stream: &CudaStream) -> Result<()> {
    if !src_host.is_empty() {
        cuda_check(
            cudaMemcpyAsync(
                dst_dev,
                src_host.as_ptr().cast(),
                src_host.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream.raw.0,
            ),
            "cudaMemcpyAsync H2D raw",
        )?;
    }
    Ok(())
}

pub unsafe fn copy_d2h_raw_async(dst_host: &mut [u8], src_dev: *const c_void, stream: &CudaStream) -> Result<()> {
    if !dst_host.is_empty() {
        cuda_check(
            cudaMemcpyAsync(
                dst_host.as_mut_ptr().cast(),
                src_dev,
                dst_host.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream.raw.0,
            ),
            "cudaMemcpyAsync D2H raw",
        )?;
    }
    Ok(())
}

pub unsafe fn copy_d2d_raw_async(dst_dev: *mut c_void, src_dev: *const c_void, bytes: usize, stream: &CudaStream) -> Result<()> {
    if bytes > 0 {
        cuda_check(
            cudaMemcpyAsync(
                dst_dev,
                src_dev,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                stream.raw.0,
            ),
            "cudaMemcpyAsync D2D raw",
        )?;
    }
    Ok(())
}

pub unsafe fn copy_peer_raw_async(
    dst_dev: *mut c_void,
    dst_device: i32,
    src_dev: *const c_void,
    src_device: i32,
    bytes: usize,
    stream: &CudaStream,
) -> Result<()> {
    if bytes > 0 {
        cuda_check(
            cudaMemcpyPeerAsync(dst_dev, dst_device, src_dev, src_device, bytes, stream.raw.0),
            "cudaMemcpyPeerAsync raw",
        )?;
    }
    Ok(())
}

pub unsafe fn zero_raw_async(dst_dev: *mut c_void, bytes: usize, stream: &CudaStream) -> Result<()> {
    if bytes > 0 {
        cuda_check(
            cudaMemsetAsync(dst_dev, 0, bytes, stream.raw.0),
            "cudaMemsetAsync raw",
        )?;
    }
    Ok(())
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudaFree(self.ptr) };
        }
    }
}

pub struct PinnedMappedBuffer<T> {
    host_ptr: NonNull<T>,
    dev_ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}
unsafe impl<T: Send> Send for PinnedMappedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedMappedBuffer<T> {}

impl<T: Default + Copy> PinnedMappedBuffer<T> {
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = (len.max(1)) * std::mem::size_of::<T>();
        let mut host_raw = ptr::null_mut();
        cuda_check(
            unsafe {
                cudaHostAlloc(
                    &mut host_raw,
                    bytes,
                    CUDA_HOST_ALLOC_PORTABLE | CUDA_HOST_ALLOC_MAPPED,
                )
            },
            "cudaHostAlloc",
        )?;
        let mut dev_ptr = ptr::null_mut();
        cuda_check(
            unsafe { cudaHostGetDevicePointer(&mut dev_ptr, host_raw, 0) },
            "cudaHostGetDevicePointer",
        )?;
        let host_ptr = NonNull::new(host_raw.cast::<T>()).expect("non-null host ptr");
        unsafe {
            for i in 0..len {
                host_ptr.as_ptr().add(i).write(T::default());
            }
        }
        Ok(Self {
            host_ptr,
            dev_ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.host_ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr.as_ptr(), self.len) }
    }

    pub fn host_ptr(&self) -> *mut T {
        self.host_ptr.as_ptr()
    }

    pub fn dev_ptr(&self) -> *mut c_void {
        self.dev_ptr
    }

    pub fn dev_ptr_for_device(&self, device: i32) -> Result<*mut c_void> {
        set_device(device)?;
        let mut p = ptr::null_mut();
        cuda_check(
            unsafe { cudaHostGetDevicePointer(&mut p, self.host_ptr.as_ptr().cast(), 0) },
            "cudaHostGetDevicePointer",
        )?;
        Ok(p)
    }
}

impl<T> Drop for PinnedMappedBuffer<T> {
    fn drop(&mut self) {
        let _ = unsafe { cudaFreeHost(self.host_ptr.as_ptr().cast()) };
    }
}

/// Page-locked host buffer (`cudaHostAllocPortable`, standard WB DRAM) for zero-copy PCIe DMA and full CPU AVX-512 speed.
pub struct PinnedHostBuffer {
    ptr: NonNull<u8>,
    bytes: usize,
}
unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let sz = bytes.max(1);
        let mut raw = ptr::null_mut();
        cuda_check(
            unsafe { cudaHostAlloc(&mut raw, sz, CUDA_HOST_ALLOC_PORTABLE) },
            "cudaHostAlloc portable",
        )?;
        Ok(Self {
            ptr: NonNull::new(raw.cast::<u8>()).expect("non-null host ptr"),
            bytes: sz,
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.bytes) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.bytes) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        let _ = unsafe { cudaFreeHost(self.ptr.as_ptr().cast()) };
    }
}

pub struct RegisteredHostMemory {
    ptr: *mut c_void,
    dev_ptr: *mut c_void,
}
unsafe impl Send for RegisteredHostMemory {}
unsafe impl Sync for RegisteredHostMemory {}

impl RegisteredHostMemory {
    pub fn register(ptr: *mut c_void, bytes: usize) -> Result<Self> {
        cuda_check(
            unsafe {
                cudaHostRegister(
                    ptr,
                    bytes,
                    CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED,
                )
            },
            "cudaHostRegister",
        )?;
        let mut dev_ptr = ptr::null_mut();
        cuda_check(
            unsafe { cudaHostGetDevicePointer(&mut dev_ptr, ptr, 0) },
            "cudaHostGetDevicePointer",
        )?;
        Ok(Self { ptr, dev_ptr })
    }

    pub fn dev_ptr(&self) -> *mut c_void {
        self.dev_ptr
    }
}

impl Drop for RegisteredHostMemory {
    fn drop(&mut self) {
        let _ = unsafe { cudaHostUnregister(self.ptr) };
    }
}

fn should_advise_hugepages(alloc_bytes: usize) -> bool {
    let Ok(text) = std::fs::read_to_string("/proc/buddyinfo") else {
        return false;
    };
    let mut order9plus_bytes = 0usize;
    for line in text.lines() {
        if let Some((_, rhs)) = line.split_once("zone") {
            let mut parts = rhs.split_whitespace();
            let _zone_name = parts.next();
            for (order, tok) in parts.enumerate() {
                if order >= 9 {
                    if let Ok(cnt) = tok.parse::<usize>() {
                        order9plus_bytes =
                            order9plus_bytes.saturating_add(cnt.saturating_mul(4096usize << order));
                    }
                }
            }
        }
    }
    order9plus_bytes >= alloc_bytes.saturating_add(1usize << 30)
}

/// Anonymous `mmap` buffer (optional `MADV_HUGEPAGE`, pre-faulted, then `cudaHostRegister(PORTABLE | MAPPED)`)
/// for high-bandwidth CPU AVX-512 reads and zero-copy GPU PCIe `uint4` reads.
pub struct RegisteredMappedHostBuffer {
    ptr: NonNull<u8>,
    bytes: usize,
}
unsafe impl Send for RegisteredMappedHostBuffer {}
unsafe impl Sync for RegisteredMappedHostBuffer {}

impl RegisteredMappedHostBuffer {
    pub fn alloc(bytes: usize) -> Result<Self> {
        let aligned = bytes
            .max(4096)
            .checked_add(4095)
            .ok_or_else(|| anyhow::anyhow!("RegisteredMappedHostBuffer size overflow"))?
            & !4095;
        let raw = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED || raw.is_null() {
            bail!(
                "mmap({aligned}) failed for RegisteredMappedHostBuffer: {}",
                std::io::Error::last_os_error()
            );
        }
        if should_advise_hugepages(aligned) {
            unsafe {
                libc::madvise(raw, aligned, libc::MADV_HUGEPAGE);
            }
        }
        // Fault all 4K pages before cudaHostRegister so page faults do not hold the CUDA driver lock.
        unsafe {
            let p = raw.cast::<u8>();
            let mut off = 0usize;
            while off < aligned {
                std::ptr::write_volatile(p.add(off), 0);
                off += 4096;
            }
        }
        if let Err(e) = cuda_check(
            unsafe {
                cudaHostRegister(
                    raw,
                    aligned,
                    CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED,
                )
            },
            "cudaHostRegister portable|mapped",
        ) {
            unsafe {
                libc::munmap(raw, aligned);
            }
            return Err(e);
        }
        Ok(Self {
            ptr: NonNull::new(raw.cast::<u8>()).expect("non-null mmap ptr"),
            bytes: aligned,
        })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn dev_ptr_for_device(&self, dev: i32) -> Result<u64> {
        set_device(dev)?;
        let mut p = ptr::null_mut();
        cuda_check(
            unsafe { cudaHostGetDevicePointer(&mut p, self.ptr.as_ptr().cast(), 0) },
            "cudaHostGetDevicePointer",
        )?;
        Ok(p as u64)
    }
}

impl Drop for RegisteredMappedHostBuffer {
    fn drop(&mut self) {
        let _ = unsafe { cudaHostUnregister(self.ptr.as_ptr().cast()) };
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.bytes) };
    }
}

pub struct CudaGraphExec {
    exec: CudaGraphExecHandle,
}
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    pub fn capture<F>(stream: &CudaStream, f: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        set_device(stream.device)?;
        cuda_check(
            unsafe { cudaStreamBeginCapture(stream.raw.0, CUDA_STREAM_CAPTURE_MODE_GLOBAL) },
            "cudaStreamBeginCapture",
        )?;
        let body_res = f();
        let mut graph = ptr::null_mut();
        let end_res = cuda_check(
            unsafe { cudaStreamEndCapture(stream.raw.0, &mut graph) },
            "cudaStreamEndCapture",
        );
        if let Err(e) = body_res {
            if !graph.is_null() {
                let _ = unsafe { cudaGraphDestroy(graph) };
            }
            return Err(e);
        }
        end_res?;
        let mut exec = ptr::null_mut();
        let inst_res = cuda_check(
            unsafe { cudaGraphInstantiateWithFlags(&mut exec, graph, 0) },
            "cudaGraphInstantiateWithFlags",
        );
        let _ = unsafe { cudaGraphDestroy(graph) };
        inst_res?;
        Ok(Self {
            exec: CudaGraphExecHandle(exec),
        })
    }

    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        cuda_check(
            unsafe { cudaGraphLaunch(self.exec.0, stream.raw.0) },
            "cudaGraphLaunch",
        )
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.exec.0.is_null() {
            let _ = unsafe { cudaGraphExecDestroy(self.exec.0) };
        }
    }
}
