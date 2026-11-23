use ampllm_gguf::{Hy4VisionConfig, Mm3VisionConfig, Qwen4VisionConfig};
use anyhow::{bail, Context, Result};
use base64::Engine as _;
use image::imageops::FilterType;
use image::RgbImage;

const MAX_ASPECT_RATIO: f64 = 200.0;

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub n_tokens: usize,
}

/// Resize target dimensions to multiples of `factor` within `[min_pixels, max_pixels]`,
/// matching Hy4/Qwen2-VL `smart_resize` semantics.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        bail!("invalid image dimensions {width}x{height} with factor {factor}");
    }
    let ratio = (height.max(width) as f64) / (height.min(width) as f64);
    if ratio > MAX_ASPECT_RATIO {
        bail!("image aspect ratio {ratio:.1} exceeds maximum {MAX_ASPECT_RATIO}:1");
    }
    let f = factor as f64;
    let mut h_bar = ((height as f64 / f).round() as usize * factor).max(factor);
    let mut w_bar = ((width as f64 / f).round() as usize * factor).max(factor);

    if h_bar.saturating_mul(w_bar) > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / f).floor() as usize * factor).max(factor);
        w_bar = ((width as f64 / beta / f).floor() as usize * factor).max(factor);
    } else if h_bar.saturating_mul(w_bar) < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / f).ceil() as usize * factor).max(factor);
        w_bar = ((width as f64 * beta / f).ceil() as usize * factor).max(factor);
    }

    while h_bar.saturating_mul(w_bar) > max_pixels && (h_bar > factor || w_bar > factor) {
        if h_bar >= w_bar && h_bar > factor {
            h_bar -= factor;
        } else if w_bar > factor {
            w_bar -= factor;
        } else {
            break;
        }
    }
    Ok((h_bar, w_bar))
}

/// Load image bytes from a `data:...;base64,...` URI, `file://` URI, or local filesystem path.
pub fn load_image_source(source: &str) -> Result<Vec<u8>> {
    let trimmed = source.trim();
    if let Some(rest) = trimmed.strip_prefix("data:") {
        let Some((header, payload)) = rest.split_once(',') else {
            bail!("malformed data URL image source (missing comma separator)");
        };
        if !header.contains(";base64") {
            bail!("data URL image source must be base64-encoded");
        }
        let compact: String = payload.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        return base64::engine::general_purpose::STANDARD
            .decode(compact.as_bytes())
            .context("invalid base64 image payload");
    }
    if let Some(path) = trimmed.strip_prefix("file://") {
        return std::fs::read(path).with_context(|| format!("reading image file {path}"));
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        bail!(
            "HTTP/HTTPS image URLs are not fetched by ampllm; pass a base64 data: URL or local file path"
        );
    }
    std::fs::read(trimmed).with_context(|| format!("reading image file {trimmed}"))
}

fn decode_and_prepare_with(
    bytes: &[u8],
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
    tokens_fn: impl FnOnce(usize, usize) -> usize,
) -> Result<PreparedImage> {
    let dyn_img = image::load_from_memory(bytes).context("decoding image bytes")?;
    let rgba = dyn_img.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);

    let mut rgb_raw = vec![0u8; w * h * 3];
    for (dst, px) in rgb_raw.chunks_exact_mut(3).zip(rgba.pixels()) {
        let [r, g, b, a] = px.0;
        if a == 255 {
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        } else {
            let af = a as u16;
            let inv = 255u16 - af;
            dst[0] = ((r as u16 * af + 255 * inv + 127) / 255) as u8;
            dst[1] = ((g as u16 * af + 255 * inv + 127) / 255) as u8;
            dst[2] = ((b as u16 * af + 255 * inv + 127) / 255) as u8;
        }
    }

    let (h_bar, w_bar) = smart_resize(h, w, factor, min_pixels, max_pixels)?;
    let rgb_img =
        RgbImage::from_raw(w as u32, h as u32, rgb_raw).context("RGB image buffer geometry")?;
    let final_rgb = if w_bar != w || h_bar != h {
        image::imageops::resize(&rgb_img, w_bar as u32, h_bar as u32, FilterType::Lanczos3)
            .into_raw()
    } else {
        rgb_img.into_raw()
    };

    let n_tokens = tokens_fn(w_bar, h_bar);
    anyhow::ensure!(
        n_tokens > 0,
        "prepared image {w_bar}x{h_bar} produced 0 vision tokens"
    );

    Ok(PreparedImage {
        rgb: final_rgb,
        width: w_bar,
        height: h_bar,
        n_tokens,
    })
}

/// Decode encoded image bytes (PNG/JPEG/GIF/BMP/WebP), alpha-composite transparency
/// onto white, and Lanczos3-resize to `Hy4VisionConfig` grid multiples.
pub fn decode_and_prepare_image(bytes: &[u8], cfg: &Hy4VisionConfig) -> Result<PreparedImage> {
    decode_and_prepare_with(bytes, cfg.factor(), cfg.min_pixels, cfg.max_pixels, |w, h| {
        cfg.image_tokens(w, h)
    })
}

/// Decode encoded image bytes (PNG/JPEG/GIF/BMP/WebP), alpha-composite transparency
/// onto white, and Lanczos3-resize to `Qwen4VisionConfig` grid multiples.
pub fn decode_and_prepare_qwen4_image(
    bytes: &[u8],
    cfg: &Qwen4VisionConfig,
) -> Result<PreparedImage> {
    decode_and_prepare_with(bytes, cfg.factor(), cfg.min_pixels, cfg.max_pixels, |w, h| {
        cfg.image_tokens(w, h)
    })
}

fn expand_image_tokens_with(
    ids: &[u32],
    pics: &[PreparedImage],
    start_id: u32,
    token_id: u32,
    end_id: u32,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let extra: usize = pics.iter().map(|p| p.n_tokens.saturating_sub(1)).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut starts = Vec::with_capacity(pics.len());
    let mut k = 0usize;

    for (i, &t) in ids.iter().enumerate() {
        if t == token_id {
            if k >= pics.len()
                || i == 0
                || ids[i - 1] != start_id
                || i + 1 >= ids.len()
                || ids[i + 1] != end_id
            {
                bail!("prompt image placeholders and image attachments do not match");
            }
            starts.push(out.len());
            out.extend(std::iter::repeat_n(token_id, pics[k].n_tokens));
            k += 1;
        } else {
            out.push(t);
        }
    }
    if k != pics.len() {
        bail!(
            "prompt had {k} image placeholders but {} images were supplied",
            pics.len()
        );
    }
    Ok((out, starts))
}

/// Replace each placeholder `<image_start><image_pad><image_end>` in `ids` with `pics[k].n_tokens`
/// copies of `cfg.image_token_id` and return `(expanded_ids, image_start_positions)`.
pub fn expand_hy4_image_tokens(
    ids: &[u32],
    pics: &[PreparedImage],
    cfg: &Hy4VisionConfig,
) -> Result<(Vec<u32>, Vec<usize>)> {
    expand_image_tokens_with(
        ids,
        pics,
        cfg.image_start_id,
        cfg.image_token_id,
        cfg.image_end_id,
    )
}

/// Replace each `<|vision_start|><|image_pad|><|vision_end|>` in `ids` with `pics[k].n_tokens`
/// copies of `cfg.image_token_id` and return `(expanded_ids, image_start_positions)`.
pub fn expand_qwen4_image_tokens(
    ids: &[u32],
    pics: &[PreparedImage],
    cfg: &Qwen4VisionConfig,
) -> Result<(Vec<u32>, Vec<usize>)> {
    expand_image_tokens_with(
        ids,
        pics,
        cfg.vision_start_id,
        cfg.image_token_id,
        cfg.vision_end_id,
    )
}

/// Load, preprocess, expand image placeholder tokens, and encode all images with GPU 0 vision tower.
pub fn prepare_and_encode_hy4_images(
    engine: &mut ampllm_engine::AnyEngine,
    prompt_ids: &[u32],
    image_sources: &[String],
) -> Result<(Vec<u32>, Vec<ampllm_engine::Hy4ImageSpan>)> {
    let Some(vcfg) = engine.hy4_vision_config().cloned() else {
        bail!("model was loaded without mmproj vision weights");
    };
    let mut pics = Vec::with_capacity(image_sources.len());
    for src in image_sources {
        let raw = load_image_source(src)?;
        pics.push(decode_and_prepare_image(&raw, &vcfg)?);
    }
    let (expanded_ids, starts) = expand_hy4_image_tokens(prompt_ids, &pics, &vcfg)?;
    let mut spans = Vec::with_capacity(pics.len());
    for (pic, start_pos) in pics.iter().zip(starts) {
        let rows = engine.encode_image(&pic.rgb, pic.width, pic.height)?;
        spans.push(ampllm_engine::Hy4ImageSpan { start_pos, rows });
    }
    Ok((expanded_ids, spans))
}

/// Load, preprocess, expand `<|image_pad|>` tokens, and encode Qwen4 images on GPU 0 vision tower.
pub fn prepare_and_encode_qwen4_images(
    engine: &mut ampllm_engine::AnyEngine,
    prompt_ids: &[u32],
    image_sources: &[String],
) -> Result<(Vec<u32>, Vec<ampllm_engine::Qwen4ImageSpan>)> {
    let Some(vcfg) = engine.qwen4_vision_config().cloned() else {
        bail!("model was loaded without mmproj vision weights");
    };
    let mut pics = Vec::with_capacity(image_sources.len());
    for src in image_sources {
        let raw = load_image_source(src)?;
        pics.push(decode_and_prepare_qwen4_image(&raw, &vcfg)?);
    }
    let (expanded_ids, starts) = expand_qwen4_image_tokens(prompt_ids, &pics, &vcfg)?;
    let mut spans = Vec::with_capacity(pics.len());
    for (pic, start_pos) in pics.iter().zip(starts) {
        let (nx, ny, rows) = engine.encode_qwen4_image(&pic.rgb, pic.width, pic.height)?;
        spans.push(ampllm_engine::Qwen4ImageSpan {
            start_pos,
            nx,
            ny,
            rows,
        });
    }
    Ok((expanded_ids, spans))
}

/// Decode encoded image bytes (PNG/JPEG/GIF/BMP/WebP), alpha-composite transparency
/// onto white, and Lanczos3-resize to `Mm3VisionConfig` grid multiples (`patch_size * spatial_merge_size`).
pub fn decode_and_prepare_mm3_image(bytes: &[u8], cfg: &Mm3VisionConfig) -> Result<PreparedImage> {
    let factor = cfg.patch_size * cfg.spatial_merge_size;
    decode_and_prepare_with(bytes, factor, cfg.min_pixels, cfg.max_pixels, |w, h| {
        (w / factor) * (h / factor)
    })
}

/// Replace each `cfg.image_token_id` placeholder (or `<start><image><end>` triple) in `ids`
/// with `[image_start_token_id, image_token_id * n_tokens, image_end_token_id]`,
/// returning `(expanded_ids, image_start_positions)` where each start points to the first `image_token_id`.
pub fn expand_mm3_image_tokens(
    ids: &[u32],
    pics: &[PreparedImage],
    cfg: &Mm3VisionConfig,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let extra: usize = pics.iter().map(|p| p.n_tokens + 2).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut starts = Vec::with_capacity(pics.len());
    let mut k = 0usize;

    for (i, &t) in ids.iter().enumerate() {
        if t == cfg.image_token_id {
            if k >= pics.len() {
                bail!("prompt image placeholders and image attachments do not match");
            }
            let has_start = i > 0 && ids[i - 1] == cfg.image_start_token_id;
            let has_end = i + 1 < ids.len() && ids[i + 1] == cfg.image_end_token_id;
            if has_start != has_end {
                bail!("prompt image start/end tags are unbalanced around placeholder {i}");
            }
            if !has_start {
                out.push(cfg.image_start_token_id);
            }
            starts.push(out.len());
            out.extend(std::iter::repeat_n(cfg.image_token_id, pics[k].n_tokens));
            if !has_end {
                out.push(cfg.image_end_token_id);
            }
            k += 1;
        } else {
            out.push(t);
        }
    }
    if k != pics.len() {
        bail!(
            "prompt had {k} image placeholders but {} images were supplied",
            pics.len()
        );
    }
    Ok((out, starts))
}

/// Load, preprocess, expand `]<]image[>[` placeholders, and encode MM3 images on GPU 0 vision tower.
pub fn prepare_and_encode_mm3_images(
    engine: &mut ampllm_engine::AnyEngine,
    prompt_ids: &[u32],
    image_sources: &[String],
) -> Result<(Vec<u32>, Vec<ampllm_engine::Mm3ImageSpan>)> {
    let Some(vcfg) = engine.mm3_vision_config().cloned() else {
        bail!("model was loaded without MM3 vision weights");
    };
    let mut pics = Vec::with_capacity(image_sources.len());
    for src in image_sources {
        let raw = load_image_source(src)?;
        pics.push(decode_and_prepare_mm3_image(&raw, &vcfg)?);
    }
    let (expanded_ids, starts) = expand_mm3_image_tokens(prompt_ids, &pics, &vcfg)?;
    let mut spans = Vec::with_capacity(pics.len());
    for (pic, start_pos) in pics.iter().zip(starts) {
        let rows = engine.encode_image(&pic.rgb, pic.width, pic.height)?;
        spans.push(ampllm_engine::Mm3ImageSpan {
            start_pos,
            n_tokens: pic.n_tokens,
            rows,
        });
    }
    Ok((expanded_ids, spans))
}
