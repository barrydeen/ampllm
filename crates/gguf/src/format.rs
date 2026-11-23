use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid GGUF magic")]
    InvalidMagic,
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("unsupported GGUF value type {0}")]
    UnsupportedValueType(u32),
    #[error("unsupported GGML tensor type {0}")]
    UnsupportedGgmlType(u32),
    #[error("truncated GGUF file at offset {0}")]
    UnexpectedEof(usize),
    #[error("invalid UTF-8 string in GGUF")]
    InvalidUtf8,
    #[error("missing metadata key '{0}'")]
    MissingMetadata(String),
    #[error("metadata key '{key}' has unexpected type (expected {expected})")]
    MetadataTypeMismatch { key: String, expected: &'static str },
    #[error("missing tensor '{0}'")]
    MissingTensor(String),
    #[error("tensor '{0}' offset/size out of file bounds")]
    TensorOutOfBounds(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Nvfp4 = 40,
    Q2_0 = 42,
    F8E4m3 = 43,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            40 => Ok(Self::Nvfp4),
            42 => Ok(Self::Q2_0),
            43 => Ok(Self::F8E4m3),
            other => Err(GgufError::UnsupportedGgmlType(other)),
        }
    }

    pub const fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 | Self::F8E4m3 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 | Self::Iq4Nl => 32,
            Self::Nvfp4 | Self::Q2_0 => 64,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M => 256,
        }
    }

    pub const fn type_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::I8 | Self::F8E4m3 => 1,
            Self::I64 | Self::F64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq4Nl => 18,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq4Xs => 136,
            Self::Iq1M => 56,
            Self::Nvfp4 => 36,
            Self::Q2_0 => 18,
        }
    }

    pub fn row_bytes(self, ne0: usize) -> usize {
        let bs = self.block_size();
        assert!(ne0 % bs == 0, "ne0={ne0} not divisible by block_size={bs} for {self:?}");
        (ne0 / bs) * self.type_size()
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::U16(v) => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::U32(v) => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::U64(v) => Some(v),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U8(v) => Some(v as i64),
            Self::I8(v) => Some(v as i64),
            Self::U16(v) => Some(v as i64),
            Self::I16(v) => Some(v as i64),
            Self::U32(v) => Some(v as i64),
            Self::I32(v) => Some(v as i64),
            Self::U64(v) => i64::try_from(v).ok(),
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            Self::U32(v) => Some(v as f32),
            Self::I32(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub rel_offset: u64,
    pub shard_idx: usize,
    pub data_offset: usize,
    pub byte_len: usize,
}

impl GgufTensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Inkling2,
    Glm5,
    Hy4,
    Qwen4Exp,
    MinimaxM3,
}

pub struct GgufFile {
    mmaps: Vec<Mmap>,
    pub shard_paths: Vec<PathBuf>,
    pub version: u32,
    pub alignment: usize,
    pub data_start: usize,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    tensor_index: HashMap<String, usize>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::UnexpectedEof(self.pos))?;
        if end > self.bytes.len() {
            return Err(GgufError::UnexpectedEof(self.pos));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let raw = self.read_exact(len)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| GgufError::InvalidUtf8)
    }

    fn read_value(&mut self, vtype: u32) -> Result<GgufValue, GgufError> {
        match vtype {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::Str(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut vec = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    vec.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(vec))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(GgufError::UnsupportedValueType(other)),
        }
    }
}

fn align_up(offset: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (offset + mask) & !mask
}

/// Read only the KV metadata map of a GGUF file without touching or retaining tensor data.
pub fn read_gguf_metadata_only(path: &Path) -> Result<HashMap<String, GgufValue>, GgufError> {
    let file = File::open(path)?;
    // SAFETY: Read-only mapping of an immutable GGUF weight file.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mut c = Cursor::new(&mmap);
    let magic = c.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = c.read_u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let _n_tensors = c.read_u64()? as usize;
    let n_kv = c.read_u64()? as usize;
    let mut metadata = HashMap::with_capacity(n_kv);
    for _ in 0..n_kv {
        let key = c.read_string()?;
        let vtype = c.read_u32()?;
        let val = c.read_value(vtype)?;
        metadata.insert(key, val);
    }
    Ok(metadata)
}

fn parse_split_suffix(fname: &str) -> Option<(&str, usize, usize)> {
    let stem = fname.strip_suffix(".gguf")?;
    let (prefix, rest) = stem.rsplit_once("-of-")?;
    if rest.len() != 5 || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (base, idx_str) = prefix.rsplit_once('-')?;
    if idx_str.len() != 5 || !idx_str.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let idx: usize = idx_str.parse().ok()?;
    let total: usize = rest.parse().ok()?;
    Some((base, idx, total))
}

fn discover_shard_paths(path: &Path) -> Vec<std::path::PathBuf> {
    let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
        return vec![path.to_path_buf()];
    };
    // Match `*-00001-of-00003` (GGUF split naming convention)
    if let Some((base, _idx, total)) = parse_split_suffix(fname) {
        if total > 1 {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let mut exact = Vec::with_capacity(total);
            for s in 1..=total {
                exact.push(parent.join(format!("{base}-{s:05}-of-{total:05}.gguf")));
            }
            if exact.iter().all(|p| p.exists()) {
                return exact;
            }
            // Cross-quant split discovery (e.g. Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002 + Hy4-Flash-Maya-XS-IQ2_XXS-00002-of-00002):
            // inspect sibling `.gguf` headers in the same folder for matching `split.count` and `split.no`.
            let first_arch = read_gguf_metadata_only(path)
                .ok()
                .and_then(|m| m.get("general.architecture").and_then(|v| v.as_str()).map(str::to_owned));
            let mut by_slot: Vec<Option<PathBuf>> = vec![None; total];
            if let Ok(entries) = std::fs::read_dir(parent) {
                let mut candidates: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|de| de.path()))
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
                    .collect();
                candidates.sort();
                for cand in candidates {
                    let Ok(meta) = read_gguf_metadata_only(&cand) else {
                        continue;
                    };
                    let kind = meta.get("general.type").and_then(|v| v.as_str());
                    if matches!(kind, Some("mtp" | "mmproj" | "drafter")) {
                        continue;
                    }
                    let arch = meta.get("general.architecture").and_then(|v| v.as_str());
                    if let (Some(fa), Some(ca)) = (first_arch.as_deref(), arch) {
                        if fa != ca {
                            continue;
                        }
                    }
                    let split_count = meta
                        .get("split.count")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .or_else(|| {
                            cand.file_name()
                                .and_then(|s| s.to_str())
                                .and_then(parse_split_suffix)
                                .map(|(_, _, t)| t)
                        });
                    if split_count != Some(total) {
                        continue;
                    }
                    let split_no = meta
                        .get("split.no")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as usize)
                        .or_else(|| {
                            cand.file_name()
                                .and_then(|s| s.to_str())
                                .and_then(parse_split_suffix)
                                .and_then(|(_, idx, _)| idx.checked_sub(1))
                        });
                    if let Some(slot) = split_no {
                        if slot < total && by_slot[slot].is_none() {
                            by_slot[slot] = Some(cand);
                        }
                    }
                }
            }
            if by_slot.iter().all(|x| x.is_some()) {
                return by_slot.into_iter().map(|x| x.unwrap()).collect();
            }
        }
    }
    vec![path.to_path_buf()]
}

/// Auto-discover standalone Hy4 auxiliary `.gguf` files (`mtp` and `mmproj`) alongside `model_path`.
/// Pass `"none"` to disable, `"auto"` or `None` to auto-detect, or a filename / path.
pub fn discover_hy4_aux_files(
    model_path: &Path,
    mtp_arg: Option<&str>,
    mmproj_arg: Option<&str>,
) -> Result<(Option<PathBuf>, Option<PathBuf>), GgufError> {
    let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
    let resolve_explicit = |raw: &str| -> PathBuf {
        let p = PathBuf::from(raw);
        if p.is_absolute() || p.exists() {
            p
        } else {
            model_dir.join(p)
        }
    };

    let mut need_scan_mtp = false;
    let mut mtp_out: Option<PathBuf> = match mtp_arg {
        Some(s) if s.eq_ignore_ascii_case("none") => None,
        Some(s) if !s.eq_ignore_ascii_case("auto") && !s.is_empty() => {
            let p = resolve_explicit(s);
            if !p.exists() {
                return Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Hy4 MTP file not found: {}", p.display()),
                )));
            }
            Some(p)
        }
        _ => {
            need_scan_mtp = true;
            None
        }
    };

    let mut need_scan_mmproj = false;
    let mut mmproj_out: Option<PathBuf> = match mmproj_arg {
        Some(s) if s.eq_ignore_ascii_case("none") => None,
        Some(s) if !s.eq_ignore_ascii_case("auto") && !s.is_empty() => {
            let p = resolve_explicit(s);
            if !p.exists() {
                return Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Hy4 mmproj file not found: {}", p.display()),
                )));
            }
            Some(p)
        }
        _ => {
            need_scan_mmproj = true;
            None
        }
    };

    if need_scan_mtp || need_scan_mmproj {
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            let mut candidates: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|de| de.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
                .collect();
            candidates.sort();
            for cand in candidates {
                let Ok(meta) = read_gguf_metadata_only(&cand) else {
                    continue;
                };
                let kind = meta.get("general.type").and_then(|v| v.as_str());
                let arch = meta.get("general.architecture").and_then(|v| v.as_str());
                if need_scan_mtp && mtp_out.is_none() && kind == Some("mtp") && arch == Some("hy_v4_flash") {
                    mtp_out = Some(cand.clone());
                }
                if need_scan_mmproj && mmproj_out.is_none() && (kind == Some("mmproj") || arch == Some("clip")) {
                    let proj = meta.get("clip.projector_type").and_then(|v| v.as_str());
                    if proj == Some("hy_v4_flash") {
                        mmproj_out = Some(cand);
                    }
                }
            }
        }
    }

    Ok((mtp_out, mmproj_out))
}

/// Auto-discover standalone Qwen3.8-Flash-Next (`qwen4exp`) auxiliary `.gguf` files
/// (`mtp` and `mmproj`) alongside `model_path` or in parent / sibling `mtp/` directories.
pub fn discover_qwen4_aux_files(
    model_path: &Path,
    explicit_mtp: Option<&Path>,
    explicit_mmproj: Option<&Path>,
) -> Result<(Option<PathBuf>, Option<PathBuf>), GgufError> {
    let mut mtp_out = None;
    if let Some(p) = explicit_mtp {
        if !p.exists() {
            return Err(GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Qwen4 MTP file not found: {}", p.display()),
            )));
        }
        mtp_out = Some(p.to_path_buf());
    }
    let mut mmproj_out = None;
    if let Some(p) = explicit_mmproj {
        if !p.exists() {
            return Err(GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Qwen4 mmproj file not found: {}", p.display()),
            )));
        }
        mmproj_out = Some(p.to_path_buf());
    }

    if mtp_out.is_none() || mmproj_out.is_none() {
        let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
        let mut search_dirs = vec![model_dir.to_path_buf()];
        if let Some(parent) = model_dir.parent() {
            search_dirs.push(parent.to_path_buf());
            search_dirs.push(parent.join("mtp"));
            if let Some(grand) = parent.parent() {
                search_dirs.push(grand.join("mtp"));
            }
        }
        for dir in search_dirs {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            let mut cands: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|de| de.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
                .collect();
            cands.sort();
            for cand in cands {
                let lower = cand
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if mmproj_out.is_none() && lower.contains("mmproj") {
                    if let Ok(meta) = read_gguf_metadata_only(&cand) {
                        let kind = meta.get("general.type").and_then(|v| v.as_str());
                        let arch = meta.get("general.architecture").and_then(|v| v.as_str());
                        let proj = meta.get("clip.projector_type").and_then(|v| v.as_str());
                        if (kind == Some("mmproj") || arch == Some("clip"))
                            && (proj == Some("qwen3vl_merger")
                                || lower.contains("qwen")
                                || dir == model_dir)
                        {
                            mmproj_out = Some(cand.clone());
                        }
                    }
                }
                if mtp_out.is_none() && lower.contains("mtp") {
                    if let Ok(meta) = read_gguf_metadata_only(&cand) {
                        let kind = meta.get("general.type").and_then(|v| v.as_str());
                        let arch = meta.get("general.architecture").and_then(|v| v.as_str());
                        if arch.is_some_and(|a| a.starts_with("qwen4"))
                            || (kind == Some("mtp") && arch != Some("hy_v4_flash"))
                        {
                            mtp_out = Some(cand);
                        }
                    }
                }
            }
        }
    }

    Ok((mtp_out, mmproj_out))
}

/// Companion and auxiliary files for MiniMax-M3.1-Flash (`minimax-m3`).
#[derive(Debug, Clone, Default)]
pub struct Mm3AuxFiles {
    pub indexers_bin: Option<PathBuf>,
    pub companions_dir: Option<PathBuf>,
    pub vision_safetensors: Option<PathBuf>,
    pub vision_config_json: Option<PathBuf>,
    pub preprocessor_json: Option<PathBuf>,
    pub dspark_config_json: Option<PathBuf>,
    pub dspark_safetensors: Option<PathBuf>,
}

/// Auto-discover MiniMax-M3.1-Flash auxiliary files:
/// - `*-maya-indexers.bin` in `model_dir` (dense-layer Sparse Attention indexers)
/// - `MiniMax-M3.1-Flash-companions` directory (Vision SafeTensors + configs + DSpark draft weights)
pub fn discover_mm3_aux_files(
    model_path: &Path,
    explicit_companions: Option<&Path>,
    explicit_dspark_dir: Option<&Path>,
) -> Result<Mm3AuxFiles, GgufError> {
    let mut out = Mm3AuxFiles::default();
    let model_dir = model_path.parent().unwrap_or_else(|| Path::new("."));

    // 1. Discover dense-layer indexer GGUF (`*-maya-indexers.bin` or `indexers*.bin`).
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        let mut bins: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|de| de.path()))
            .filter(|p| {
                let name = p
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                name.ends_with(".bin") && name.contains("indexers")
            })
            .collect();
        bins.sort();
        if let Some(first) = bins.into_iter().next() {
            out.indexers_bin = Some(first);
        }
    }

    // 2. Discover companions directory.
    let mut comp_candidates = Vec::new();
    if let Some(p) = explicit_companions {
        if !p.exists() {
            return Err(GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("MM3 companions directory not found: {}", p.display()),
            )));
        }
        comp_candidates.push(p.to_path_buf());
    } else {
        comp_candidates.push(model_dir.to_path_buf());
        comp_candidates.push(model_dir.join("companions"));
        comp_candidates.push(model_dir.join("MiniMax-M3.1-Flash-companions"));
        if let Some(parent) = model_dir.parent() {
            comp_candidates.push(parent.join("MiniMax-M3.1-Flash-companions"));
            if let Ok(entries) = std::fs::read_dir(parent) {
                let mut extras: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok().map(|de| de.path()))
                    .filter(|p| {
                        p.is_dir()
                            && p.file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_ascii_lowercase()
                                .contains("companions")
                    })
                    .collect();
                extras.sort();
                comp_candidates.extend(extras);
            }
        }
    }

    for dir in comp_candidates {
        if !dir.is_dir() {
            continue;
        }
        let cfg = dir.join("config.json");
        let vis_single = dir.join("model-vision-00001-of-00001.safetensors");
        let has_vis = if vis_single.exists() {
            Some(vis_single)
        } else {
            std::fs::read_dir(&dir).ok().and_then(|es| {
                let mut m: Vec<PathBuf> = es
                    .filter_map(|e| e.ok().map(|de| de.path()))
                    .filter(|p| {
                        let s = p.file_name().and_then(|x| x.to_str()).unwrap_or("");
                        s.starts_with("model-vision-") && s.ends_with(".safetensors")
                    })
                    .collect();
                m.sort();
                m.into_iter().next()
            })
        };
        if cfg.exists() || has_vis.is_some() || dir.join("dspark").is_dir() {
            out.companions_dir = Some(dir.clone());
            if cfg.exists() {
                out.vision_config_json = Some(cfg);
            }
            let pre = dir.join("preprocessor_config.json");
            if pre.exists() {
                out.preprocessor_json = Some(pre);
            }
            out.vision_safetensors = has_vis;
            break;
        }
    }

    // 3. Discover DSpark directory (either explicit, `<companions>/dspark`, or `<model_dir>/dspark`).
    let mut dspark_dirs = Vec::new();
    if let Some(p) = explicit_dspark_dir {
        if !p.exists() {
            return Err(GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("MM3 DSpark directory not found: {}", p.display()),
            )));
        }
        dspark_dirs.push(p.to_path_buf());
    } else {
        if let Some(ref c) = out.companions_dir {
            dspark_dirs.push(c.join("dspark"));
        }
        dspark_dirs.push(model_dir.join("dspark"));
        if let Some(parent) = model_dir.parent() {
            dspark_dirs.push(parent.join("dspark"));
        }
    }

    for dd in dspark_dirs {
        if !dd.is_dir() {
            continue;
        }
        let dcfg = dd.join("config.json");
        let idx = dd.join("model.safetensors.index.json");
        let st_path = if idx.exists() {
            Some(idx)
        } else {
            let s1 = dd.join("model-dspark-00001.safetensors");
            if s1.exists() {
                Some(s1)
            } else {
                std::fs::read_dir(&dd).ok().and_then(|es| {
                    let mut m: Vec<PathBuf> = es
                        .filter_map(|e| e.ok().map(|de| de.path()))
                        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
                        .collect();
                    m.sort();
                    m.into_iter().next()
                })
            }
        };
        if dcfg.exists() {
            out.dspark_config_json = Some(dcfg);
        }
        if st_path.is_some() {
            out.dspark_safetensors = st_path;
        }
        if out.dspark_config_json.is_some() || out.dspark_safetensors.is_some() {
            break;
        }
    }

    Ok(out)
}

struct ParsedShard {
    mmap: Mmap,
    version: u32,
    alignment: usize,
    data_start: usize,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<GgufTensorInfo>,
}

fn parse_single_shard(path: &Path, shard_idx: usize, read_metadata: bool) -> Result<ParsedShard, GgufError> {
    let file = File::open(path)?;
    // SAFETY: Read-only mapping of an immutable GGUF weight file.
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let mut c = Cursor::new(&mmap);
    let magic = c.read_exact(4)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic);
    }
    let version = c.read_u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let n_tensors = c.read_u64()? as usize;
    let n_kv = c.read_u64()? as usize;

    let mut metadata = HashMap::with_capacity(if read_metadata { n_kv } else { 0 });
    for _ in 0..n_kv {
        let key = c.read_string()?;
        let vtype = c.read_u32()?;
        let val = c.read_value(vtype)?;
        if read_metadata || key == "general.alignment" {
            metadata.insert(key, val);
        }
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
        .max(1);

    struct RawTensor {
        name: String,
        dims: Vec<u64>,
        ggml_type: GgmlType,
        rel_offset: u64,
    }
    let mut raw_tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.read_string()?;
        let n_dims = c.read_u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(c.read_u64()?);
        }
        let ggml_type = GgmlType::from_u32(c.read_u32()?)?;
        let rel_offset = c.read_u64()?;
        raw_tensors.push(RawTensor {
            name,
            dims,
            ggml_type,
            rel_offset,
        });
    }

    let data_start = align_up(c.pos, alignment);
    let mut tensors = Vec::with_capacity(raw_tensors.len());
    for rt in raw_tensors {
        let ne0 = rt.dims.first().copied().unwrap_or(1) as usize;
        let n_rows: usize = rt.dims.iter().skip(1).map(|&d| d as usize).product();
        let byte_len = rt.ggml_type.row_bytes(ne0) * n_rows;
        let data_offset = data_start
            .checked_add(rt.rel_offset as usize)
            .ok_or_else(|| GgufError::TensorOutOfBounds(rt.name.clone()))?;
        let end = data_offset
            .checked_add(byte_len)
            .ok_or_else(|| GgufError::TensorOutOfBounds(rt.name.clone()))?;
        if end > mmap.len() {
            return Err(GgufError::TensorOutOfBounds(rt.name));
        }
        tensors.push(GgufTensorInfo {
            name: rt.name,
            dims: rt.dims,
            ggml_type: rt.ggml_type,
            rel_offset: rt.rel_offset,
            shard_idx,
            data_offset,
            byte_len,
        });
    }

    Ok(ParsedShard {
        mmap,
        version,
        alignment,
        data_start,
        metadata,
        tensors,
    })
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let shard_paths = discover_shard_paths(path.as_ref());
        let first = parse_single_shard(&shard_paths[0], 0, true)?;
        let version = first.version;
        let alignment = first.alignment;
        let data_start = first.data_start;
        let metadata = first.metadata;

        let mut mmaps = Vec::with_capacity(shard_paths.len());
        mmaps.push(first.mmap);
        let mut tensors = first.tensors;

        for (idx, p) in shard_paths.iter().enumerate().skip(1) {
            let parsed = parse_single_shard(p, idx, false)?;
            mmaps.push(parsed.mmap);
            tensors.extend(parsed.tensors);
        }

        let mut tensor_index = HashMap::with_capacity(tensors.len());
        for (idx, t) in tensors.iter().enumerate() {
            tensor_index.insert(t.name.clone(), idx);
        }

        Ok(Self {
            mmaps,
            shard_paths,
            version,
            alignment,
            data_start,
            metadata,
            tensors,
            tensor_index,
        })
    }

    pub fn shard_count(&self) -> usize {
        self.mmaps.len()
    }

    /// Attach a standalone auxiliary GGUF file (such as a standalone Hy4 `mtp` block shard)
    /// into this `GgufFile`, merging its tensors and per-block metadata (e.g. `nextn_predict_layers`).
    pub fn merge_aux_shard(&mut self, path: &Path) -> Result<(), GgufError> {
        let shard_idx = self.mmaps.len();
        let parsed = parse_single_shard(path, shard_idx, true)?;

        // Merge standalone Hy4 MTP metadata if trunk did not carry blk.<n_layer> yet.
        let arch = parsed
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if parsed.metadata.get("general.type").and_then(|v| v.as_str()) == Some("mtp")
            && arch == "hy_v4_flash"
        {
            let trunk_n = self.get_u32("hy_v4_flash.block_count").unwrap_or(0);
            let m_all = parsed
                .metadata
                .get("hy_v4_flash.block_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let m_mtp = parsed
                .metadata
                .get("hy_v4_flash.nextn_predict_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            if m_mtp == 1 && m_all == trunk_n + m_mtp {
                self.metadata
                    .insert("hy_v4_flash.block_count".to_owned(), GgufValue::U32(m_all));
                self.metadata.insert(
                    "hy_v4_flash.nextn_predict_layers".to_owned(),
                    GgufValue::U32(m_mtp),
                );
                for per_key in [
                    "hy_v4_flash.attention.head_count",
                    "hy_v4_flash.swiglu_clamp_exp",
                ] {
                    if let Some(mval) = parsed.metadata.get(per_key) {
                        let tail_elem = match mval.as_array() {
                            Some(arr) => arr.get(trunk_n as usize).cloned(),
                            None => Some(mval.clone()),
                        };
                        if let Some(tail) = tail_elem {
                            match self.metadata.get_mut(per_key) {
                                Some(GgufValue::Array(ref mut arr)) => {
                                    if arr.len() == trunk_n as usize {
                                        arr.push(tail);
                                    }
                                }
                                Some(scalar) => {
                                    let s = scalar.clone();
                                    let mut arr = vec![s; trunk_n as usize];
                                    arr.push(tail);
                                    self.metadata
                                        .insert(per_key.to_owned(), GgufValue::Array(arr));
                                }
                                None => {}
                            }
                        }
                    }
                }
            }
        }

        self.mmaps.push(parsed.mmap);
        self.shard_paths.push(path.to_path_buf());
        for t in parsed.tensors {
            let idx = self.tensors.len();
            self.tensor_index.insert(t.name.clone(), idx);
            self.tensors.push(t);
        }
        Ok(())
    }

    pub fn architecture(&self) -> Result<ModelArch, GgufError> {
        let arch = self.get_str("general.architecture")?;
        if arch.starts_with("glm5") {
            Ok(ModelArch::Glm5)
        } else if arch.starts_with("inkling2") {
            Ok(ModelArch::Inkling2)
        } else if arch == "hy_v4_flash" || arch.starts_with("hy4") {
            Ok(ModelArch::Hy4)
        } else if arch == "qwen4exp" || arch == "qwen4_exp" {
            Ok(ModelArch::Qwen4Exp)
        } else if arch == "minimax-m3" || arch.starts_with("minimax_m3") || arch == "mm3" {
            Ok(ModelArch::MinimaxM3)
        } else {
            Err(GgufError::MetadataTypeMismatch {
                key: "general.architecture".to_owned(),
                expected: "inkling2*, glm5*, hy_v4_flash, qwen4exp, or minimax-m3",
            })
        }
    }

    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo, GgufError> {
        self.tensor_index
            .get(name)
            .map(|&idx| &self.tensors[idx])
            .ok_or_else(|| GgufError::MissingTensor(name.to_owned()))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self.tensor_info(name)?;
        Ok(&self.mmaps[info.shard_idx][info.data_offset..info.data_offset + info.byte_len])
    }

    pub fn tensor_f32_slice(&self, name: &str) -> Result<&[f32], GgufError> {
        let info = self.tensor_info(name)?;
        if info.ggml_type != GgmlType::F32 {
            return Err(GgufError::MetadataTypeMismatch {
                key: name.to_owned(),
                expected: "F32 tensor",
            });
        }
        let bytes = &self.mmaps[info.shard_idx][info.data_offset..info.data_offset + info.byte_len];
        Ok(bytemuck::cast_slice(bytes))
    }

    pub fn get_str(&self, key: &str) -> Result<&str, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        val.as_str().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "string",
        })
    }

    pub fn get_u32(&self, key: &str) -> Result<u32, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        val.as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| GgufError::MetadataTypeMismatch {
                key: key.to_owned(),
                expected: "u32",
            })
    }

    pub fn get_f32(&self, key: &str) -> Result<f32, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        val.as_f32().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "f32",
        })
    }

    pub fn get_f32_or_first(&self, key: &str) -> Result<f32, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        if let Some(v) = val.as_f32() {
            return Ok(v);
        }
        if let Some(arr) = val.as_array() {
            if let Some(first) = arr.first().and_then(|v| v.as_f32()) {
                return Ok(first);
            }
        }
        Err(GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "f32 or array<f32>",
        })
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        val.as_bool().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "bool",
        })
    }

    pub fn get_u32_or_first(&self, key: &str) -> Result<u32, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        if let Some(v) = val.as_u64().and_then(|x| u32::try_from(x).ok()) {
            return Ok(v);
        }
        if let Some(arr) = val.as_array() {
            if let Some(first) = arr.first().and_then(|v| v.as_u64()).and_then(|x| u32::try_from(x).ok()) {
                return Ok(first);
            }
        }
        Err(GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "u32 or array<u32>",
        })
    }

    pub fn get_u32_array(&self, key: &str) -> Result<Vec<u32>, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        let arr = val.as_array().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "array<u32>",
        })?;
        arr.iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| GgufError::MetadataTypeMismatch {
                        key: key.to_owned(),
                        expected: "u32 element",
                    })
            })
            .collect()
    }

    pub fn get_bool_array(&self, key: &str) -> Result<Vec<bool>, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        let arr = val.as_array().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "array<bool>",
        })?;
        arr.iter()
            .map(|x| {
                x.as_bool().ok_or_else(|| GgufError::MetadataTypeMismatch {
                    key: key.to_owned(),
                    expected: "bool element",
                })
            })
            .collect()
    }

    pub fn get_i32_array(&self, key: &str) -> Result<Vec<i32>, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        let arr = val.as_array().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "array<i32>",
        })?;
        arr.iter()
            .map(|x| {
                x.as_i64()
                    .and_then(|v| i32::try_from(v).ok())
                    .ok_or_else(|| GgufError::MetadataTypeMismatch {
                        key: key.to_owned(),
                        expected: "i32 element",
                    })
            })
            .collect()
    }

    pub fn get_u64_array(&self, key: &str) -> Result<Vec<u64>, GgufError> {
        let val = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_owned()))?;
        let arr = val.as_array().ok_or_else(|| GgufError::MetadataTypeMismatch {
            key: key.to_owned(),
            expected: "array<u64>",
        })?;
        arr.iter()
            .map(|x| {
                // Accept signed integers matching uint64 bit-pattern (e.g. `multiplier & 0xFFFFFFFFFFFFFFFF`).
                x.as_u64()
                    .or_else(|| x.as_i64().map(|v| v as u64))
                    .ok_or_else(|| GgufError::MetadataTypeMismatch {
                        key: key.to_owned(),
                        expected: "u64 element",
                    })
            })
            .collect()
    }

    pub fn get_f32_array_or_broadcast(&self, key: &str, len: usize, default: f32) -> Vec<f32> {
        match self.metadata.get(key) {
            None => vec![default; len],
            Some(v) => {
                if let Some(arr) = v.as_array() {
                    arr.iter()
                        .map(|x| x.as_f32().unwrap_or(default))
                        .collect()
                } else if let Some(s) = v.as_f32() {
                    vec![s; len]
                } else {
                    vec![default; len]
                }
            }
        }
    }

    pub fn get_u32_array_or_broadcast(&self, key: &str, len: usize, default: u32) -> Vec<u32> {
        match self.metadata.get(key) {
            None => vec![default; len],
            Some(v) => {
                if let Some(arr) = v.as_array() {
                    arr.iter()
                        .map(|x| x.as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(default))
                        .collect()
                } else if let Some(s) = v.as_u64().and_then(|n| u32::try_from(n).ok()) {
                    vec![s; len]
                } else {
                    vec![default; len]
                }
            }
        }
    }

    /// Return the underlying read-only byte slice of a shard's mmap.
    pub fn shard_bytes(&self, shard_idx: usize) -> &[u8] {
        &self.mmaps[shard_idx]
    }

    pub fn advise_shard_random(&self, shard_idx: usize) {
        if let Some(m) = self.mmaps.get(shard_idx) {
            let _ = m.advise(memmap2::Advice::Random);
        }
    }

    pub fn advise_shard_sequential(&self, shard_idx: usize) {
        if let Some(m) = self.mmaps.get(shard_idx) {
            unsafe {
                libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_SEQUENTIAL);
            }
        }
    }

    pub fn advise_shard_normal(&self, shard_idx: usize) {
        if let Some(m) = self.mmaps.get(shard_idx) {
            unsafe {
                libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_NORMAL);
            }
        }
    }

    pub fn advise_range_willneed(&self, shard_idx: usize, offset: usize, len: usize) {
        let Some(m) = self.mmaps.get(shard_idx) else {
            return;
        };
        if len == 0 || offset >= m.len() {
            return;
        }
        const PG: usize = 4096;
        let a0 = offset & !(PG - 1);
        let end = offset.saturating_add(len).min(m.len());
        let a1 = (end.saturating_add(PG - 1) & !(PG - 1)).min(m.len());
        let alen = a1.saturating_sub(a0);
        if alen > 0 {
            unsafe {
                libc::madvise(
                    m.as_ptr().add(a0) as *mut libc::c_void,
                    alen,
                    libc::MADV_WILLNEED,
                );
            }
        }
    }

    pub fn advise_shard_dontneed_except(&self, shard_idx: usize, keep_start: usize, keep_len: usize) {
        let Some(m) = self.mmaps.get(shard_idx) else {
            return;
        };
        let len = m.len();
        let base = m.as_ptr() as *mut libc::c_void;
        if keep_len == 0 || keep_start >= len {
            unsafe {
                libc::madvise(base, len, libc::MADV_DONTNEED);
            }
            return;
        }
        const PG: usize = 4096;
        let k0 = keep_start & !(PG - 1);
        let k1 = keep_start
            .saturating_add(keep_len)
            .saturating_add(PG - 1)
            & !(PG - 1);
        if k0 > 0 {
            unsafe {
                libc::madvise(base, k0, libc::MADV_DONTNEED);
            }
        }
        if k1 < len {
            unsafe {
                libc::madvise((base as *mut u8).add(k1) as *mut libc::c_void, len - k1, libc::MADV_DONTNEED);
            }
        }
    }
}
