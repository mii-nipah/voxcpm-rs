#![allow(dead_code)]

use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::{Error, Result};

/// Write a 32-bit float mono waveform to a 16-bit PCM WAV file.
pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let file = std::fs::File::create(path)?;
    write_wav_to(std::io::BufWriter::new(file), samples, sample_rate)
}

/// Write a 32-bit float mono waveform as 16-bit PCM WAV to any
/// `Write + Seek` sink (e.g. `Cursor<Vec<u8>>`, `BufWriter<File>`,
/// or a memory-mapped buffer). Mirrors [`write_wav`] but lets callers
/// stream the encoded bytes anywhere — useful for HTTP responses,
/// channels, sockets, or in-memory pipelines that don't want to touch
/// the filesystem.
pub fn write_wav_to<W: std::io::Write + std::io::Seek>(
    writer: W,
    samples: &[f32],
    sample_rate: u32,
) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(writer, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        writer.write_sample((clamped * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Encode a 32-bit float mono waveform as a 16-bit PCM WAV byte buffer.
/// Convenience wrapper over [`write_wav_to`] for the common
/// "give me the bytes" case (HTTP responses, channels, etc.).
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::<u8>::new());
    write_wav_to(&mut buf, samples, sample_rate)?;
    Ok(buf.into_inner())
}

/// Decode an audio file (WAV/FLAC/MP3/etc. — anything Symphonia supports) to
/// a mono `f32` PCM waveform, returning `(samples, sample_rate)`.
pub fn load_audio(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path.as_ref())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| Error::AudioDecode(e.to_string()))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| Error::AudioDecode("no default track".into()))?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioDecode(e.to_string()))?;
    let sr = codec_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecode("unknown sample rate".into()))?;

    let mut pcm: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(Error::AudioDecode(e.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(Error::AudioDecode(e.to_string())),
        };
        append_mono_f32(&decoded, &mut pcm);
    }
    Ok((pcm, sr))
}

fn append_mono_f32(buf: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| b.chan(ch)[i]),
        AudioBufferRef::F64(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| b.chan(ch)[i] as f32),
        AudioBufferRef::S16(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            b.chan(ch)[i] as f32 / i16::MAX as f32
        }),
        AudioBufferRef::S32(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            b.chan(ch)[i] as f32 / i32::MAX as f32
        }),
        AudioBufferRef::U8(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            (b.chan(ch)[i] as f32 - 128.0) / 128.0
        }),
        AudioBufferRef::S8(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            b.chan(ch)[i] as f32 / i8::MAX as f32
        }),
        AudioBufferRef::U16(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            (b.chan(ch)[i] as f32 - 32768.0) / 32768.0
        }),
        AudioBufferRef::U24(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            (b.chan(ch)[i].inner() as f32 - 8_388_608.0) / 8_388_608.0
        }),
        AudioBufferRef::S24(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            b.chan(ch)[i].inner() as f32 / 8_388_608.0
        }),
        AudioBufferRef::U32(b) => mix_to_mono(b.spec().channels.count(), b.frames(), out, |ch, i| {
            (b.chan(ch)[i] as f64 - 2_147_483_648.0) as f32 / 2_147_483_648.0
        }),
    }
}

fn mix_to_mono<F: Fn(usize, usize) -> f32>(n_ch: usize, n_frames: usize, out: &mut Vec<f32>, sample: F) {
    if n_ch == 1 {
        for i in 0..n_frames {
            out.push(sample(0, i));
        }
    } else {
        let inv = 1.0 / n_ch as f32;
        for i in 0..n_frames {
            let mut sum = 0.0f32;
            for c in 0..n_ch {
                sum += sample(c, i);
            }
            out.push(sum * inv);
        }
    }
}

/// Resample a mono f32 waveform from `from_sr` to `to_sr` using a
/// fixed-ratio sinc resampler. No-op if the rates are equal.
pub fn resample(input: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    if from_sr == to_sr {
        return Ok(input.to_vec());
    }
    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = to_sr as f64 / from_sr as f64;
    let chunk_size = 1024usize;
    let mut resampler: SincFixedIn<f32> =
        SincFixedIn::new(ratio, 2.0, params, chunk_size, 1).map_err(|e| Error::Resampler(e.to_string()))?;

    let mut out: Vec<f32> = Vec::with_capacity(((input.len() as f64) * ratio) as usize + 64);
    let mut pos = 0usize;
    while pos + chunk_size <= input.len() {
        let block: Vec<Vec<f32>> = vec![input[pos..pos + chunk_size].to_vec()];
        let res = resampler
            .process(&block, None)
            .map_err(|e| Error::Resampler(e.to_string()))?;
        out.extend_from_slice(&res[0]);
        pos += chunk_size;
    }
    if pos < input.len() {
        let mut tail = input[pos..].to_vec();
        tail.resize(chunk_size, 0.0);
        let block: Vec<Vec<f32>> = vec![tail];
        let res = resampler
            .process(&block, None)
            .map_err(|e| Error::Resampler(e.to_string()))?;
        let expected = ((input.len() - pos) as f64 * ratio).round() as usize;
        out.extend_from_slice(&res[0][..expected.min(res[0].len())]);
    }
    Ok(out)
}

/// Load `path`, downmix to mono, and resample to `target_sr`.
pub fn load_audio_as(path: impl AsRef<Path>, target_sr: u32) -> Result<Vec<f32>> {
    let (pcm, sr) = load_audio(path)?;
    resample(&pcm, sr, target_sr)
}

/// Decode an encoded audio byte buffer (WAV/FLAC/MP3/etc.) to mono `f32` PCM,
/// returning `(samples, sample_rate)`. Same format support as
/// [`load_audio`], just sourced from memory instead of a file.
pub fn load_audio_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&Hint::new(), mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| Error::AudioDecode(e.to_string()))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| Error::AudioDecode("no default track".into()))?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| Error::AudioDecode(e.to_string()))?;
    let sr = codec_params
        .sample_rate
        .ok_or_else(|| Error::AudioDecode("unknown sample rate".into()))?;

    let mut pcm: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(Error::AudioDecode(e.to_string())),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(Error::AudioDecode(e.to_string())),
        };
        append_mono_f32(&decoded, &mut pcm);
    }
    Ok((pcm, sr))
}

/// Decode an encoded audio byte buffer, downmix to mono, and resample to `target_sr`.
pub fn load_audio_bytes_as(bytes: &[u8], target_sr: u32) -> Result<Vec<f32>> {
    let (pcm, sr) = load_audio_bytes(bytes)?;
    resample(&pcm, sr, target_sr)
}
