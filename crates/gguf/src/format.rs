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
            other => Err(GgufError::UnsupportedGgmlType(other)),
        }
    }

    pub const fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 | Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 | Self::Iq4Nl => 32,
            Self::Nvfp4 => 64,
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
            Self::I8 => 1,
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

fn discover_shard_paths(path: &Path) -> Vec<std::path::PathBuf> {
    let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
        return vec![path.to_path_buf()];
    };
    let Some(stem) = fname.strip_suffix(".gguf") else {
        return vec![path.to_path_buf()];
    };
    // Match `*-00001-of-00003` (GGUF split naming convention)
    if let Some((prefix, rest)) = stem.rsplit_once("-of-") {
        if rest.len() == 5 && rest.bytes().all(|b| b.is_ascii_digit()) {
            if let Some((base, idx_str)) = prefix.rsplit_once('-') {
                if idx_str.len() == 5 && idx_str.bytes().all(|b| b.is_ascii_digit()) {
                    if let Ok(total) = rest.parse::<usize>() {
                        if total > 1 {
                            let parent = path.parent().unwrap_or_else(|| Path::new("."));
                            let mut out = Vec::with_capacity(total);
                            for s in 1..=total {
                                out.push(parent.join(format!("{base}-{s:05}-of-{total:05}.gguf")));
                            }
                            if out.iter().all(|p| p.exists()) {
                                return out;
                            }
                        }
                    }
                }
            }
        }
    }
    vec![path.to_path_buf()]
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

    pub fn architecture(&self) -> Result<ModelArch, GgufError> {
        let arch = self.get_str("general.architecture")?;
        if arch.starts_with("glm5") {
            Ok(ModelArch::Glm5)
        } else if arch.starts_with("inkling2") {
            Ok(ModelArch::Inkling2)
        } else {
            Err(GgufError::MetadataTypeMismatch {
                key: "general.architecture".to_owned(),
                expected: "inkling2* or glm5*",
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
}
