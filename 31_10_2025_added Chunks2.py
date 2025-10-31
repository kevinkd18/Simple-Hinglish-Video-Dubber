#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# COMPLETE FIX:
# - get_voice() and batch_tts() defined (fixes NameError)
# - Chunked browser uploads (large videos reliable)
# - Audio/TTS unchanged; no-skip HI speech via TTS-length rescheduling
# - Uniform video slow-down to match final audio duration (setpts)
# - Whisper lazy-load + optional faster-whisper for speed
# - Translation resilient to 429 (Retry-After + backoff + autoscaling)
# - Stage-wise timing + total time summary
# - Smart mux: stream-copy when slow-down not needed

import os
import re
import json
import uuid
import time
import asyncio
import tempfile
import logging
import warnings
import traceback
import subprocess
import threading
import random
from typing import List, Dict

import numpy as np
import soundfile as sf
import librosa
import whisper
import edge_tts

from flask import Flask, request, send_from_directory, abort, jsonify
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
try:
    from cerebras.cloud.sdk import RateLimitError  # SDK error for 429
except Exception:
    class RateLimitError(Exception):
        pass
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil

# Optional faster-whisper (for speed)
HAVE_FASTER = False
try:
    from faster_whisper import WhisperModel as FWWhisperModel
    HAVE_FASTER = True
except Exception:
    HAVE_FASTER = False

# ── CONFIG ───────────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore")

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    print("⚠️  CEREBRAS_API_KEY missing! Translation will fail!")

client = Cerebras(api_key=CEREBRAS_API_KEY) if CEREBRAS_API_KEY else None

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
TMP_DIR = "upload_tmp"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")          # OpenAI whisper
FAST_WHISPER_MODEL = os.environ.get("FAST_WHISPER_MODEL", "base") # faster-whisper
USE_FASTER_WHISPER = os.environ.get("USE_FASTER_WHISPER", "0") == "1"
TTS_CONCURRENCY = int(os.environ.get("TTS_CONCURRENCY", "10"))
ALLOWED_EXTS = {"mp4", "mov", "mkv", "webm", "avi"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hinglish-dubber")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB per chunk

# ── TTS VOICE PICKER ─────────────────────────────────────────
def get_voice() -> str:
    """
    Return Edge TTS voice short-name; override via VOICE_NAME env var.
    Example defaults: en-IN-NeerjaNeural, en-IN-PrabhatNeural, hi-IN-SwaraNeural.
    """
    return os.environ.get("VOICE_NAME", "en-IN-NeerjaNeural")

# ── WHISPER LAZY-LOAD (two backends) ────────────────────────
_whisper_lock = threading.Lock()
_whisper_model = None

_fw_lock = threading.Lock()
_fw_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                print("Loading Whisper (OpenAI)...")
                _whisper_model = whisper.load_model(WHISPER_MODEL)
                print("Ready!")
    return _whisper_model

def get_fw_model():
    global _fw_model
    if _fw_model is None:
        with _fw_lock:
            if _fw_model is None:
                print("Loading faster-whisper...")
                compute_type = os.environ.get("FW_COMPUTE", "auto")
                if compute_type == "auto":
                    compute_type = "float16" if shutil.which("nvidia-smi") else "int8"
                _fw_model = FWWhisperModel(FAST_WHISPER_MODEL, compute_type=compute_type)
                print("Ready (faster-whisper)!")
    return _fw_model

def transcribe_audio(audio_file: str) -> List[Dict]:
    # Returns segments: [{start, end, text}, ...]
    if USE_FASTER_WHISPER and HAVE_FASTER:
        model = get_fw_model()
        segments, info = model.transcribe(audio_file, language="en", vad_filter=True)
        out = []
        for seg in segments:
            out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
        return out
    else:
        model = get_whisper_model()
        result = model.transcribe(audio_file, language='en')
        segs = result.get("segments", [])
        out = []
        for s in segs:
            out.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip()
            })
        return out

# ── FFPROBE / FFMPEG UTILS ──────────────────────────────────
def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def get_media_duration(path: str) -> float:
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
        return float(out.stdout.decode().strip())
    except:
        return 0.0

def get_video_duration(path: str) -> float:
    return get_media_duration(path)

def extract_audio_ffmpeg(video_path, audio_path):
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
           '-ar', '16000', '-ac', '1', '-y', audio_path]
    return run_ffmpeg(cmd)

# ── TRANSLATION (429-resilient) ─────────────────────────────
_translate_lock = threading.Semaphore(1)

def _sleep_backoff(attempt: int, retry_after_s: float | None, base: float = 1.0, cap: float = 30.0):
    if retry_after_s is not None and retry_after_s > 0:
        time.sleep(retry_after_s)
        return
    delay = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, min(1.0, delay * 0.25))
    time.sleep(delay + jitter)

def _parse_retry_after_from_exc(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            if ra:
                return float(ra)
        except:
            return None
    return None

def _call_cerebras_with_retry(messages, temperature=0.25, max_completion_tokens=1000, max_retries=6):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
        except RateLimitError as e:
            ra = _parse_retry_after_from_exc(e)
            logger.warning(f"429 RateLimited, attempt {attempt+1}/{max_retries}, Retry-After={ra}")
            _sleep_backoff(attempt, ra)
            last_err = e
            continue
        except Exception as e:
            last_err = e
            _sleep_backoff(attempt, None, base=0.5, cap=5.0)
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Unknown translation failure")

def translate_to_hinglish_sync(texts: List[str]) -> List[str]:
    if not texts or not client:
        print("⚠️  No client or texts")
        return texts

    cache: dict[str, str] = {}
    results: list[str] = []

    # Dedup
    uniq_map: dict[str, int] = {}
    uniq_list: list[str] = []
    for t in texts:
        if t in cache:
            continue
        if t not in uniq_map:
            uniq_map[t] = 1
            uniq_list.append(t)

    batch_size = 5
    i = 0
    with _translate_lock:
        while i < len(uniq_list):
            batch = uniq_list[i:i+batch_size]
            text_input = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
            prompt = f"""Convert these English sentences to Hinglish (mix of Hindi and English in Roman script).
Return ONLY a JSON array like ["hinglish1","hinglish2",...], same length/order as input:

{text_input}"""
            messages = [
                {"role": "system", "content": "You convert English to Hinglish. Always return valid JSON arrays."},
                {"role": "user", "content": prompt}
            ]

            try:
                resp = _call_cerebras_with_retry(messages, temperature=0.25, max_completion_tokens=1000)
                content = resp.choices[0].message.content
                m = re.search(r'\[.*?\]', content, re.DOTALL)
                if not m:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        continue
                    for t in batch:
                        cache[t] = t
                    i += len(batch)
                    continue
                arr = json.loads(m.group(0))
                if not isinstance(arr, list) or len(arr) != len(batch):
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        continue
                    for t in batch:
                        cache[t] = t
                    i += len(batch)
                    continue
                for t, h in zip(batch, arr):
                    cache[t] = h if isinstance(h, str) and h.strip() else t
                i += len(batch)
                time.sleep(0.4)
                if batch_size < 5:
                    batch_size += 1
            except RateLimitError as e:
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                else:
                    ra = _parse_retry_after_from_exc(e)
                    _sleep_backoff(3, ra, base=2.0, cap=60.0)
                continue
            except Exception:
                for t in batch:
                    cache[t] = t
                i += len(batch)
                continue

    for t in texts:
        results.append(cache.get(t, t))
    if len(results) != len(texts):
        print(f"⚠️  Result length mismatch! Expected {len(texts)}, got {len(results)}")
        return texts
    return results

# ── TTS GENERATION (fixed and present) ──────────────────────
async def gen_speech(text: str, voice: str, output: str):
    if not text or not text.strip():
        return False
    try:
        clean = re.sub(r'[^\w\s.,!?-]', '', text).strip()
        comm = edge_tts.Communicate(text=clean, voice=voice)
        await comm.save(output)
        return os.path.exists(output)
    except:
        return False

async def batch_tts(texts: List[str], files: List[str], voice: str):
    sem = asyncio.Semaphore(TTS_CONCURRENCY)
    async def gen_one(txt, f):
        async with sem:
            return await gen_speech(txt, voice, f)
    tasks = [gen_one(t, f) for t, f in zip(texts, files) if t and t.strip()]
    return await asyncio.gather(*tasks, return_exceptions=True)

# ── AUDIO ASSEMBLY / VIDEO SLOWDOWN / MERGE ─────────────────
def reschedule_segments_by_tts(segments: List[Dict], tts_files: List[str]) -> (List[Dict], float):
    if not segments:
        return [], 0.0
    t = max(0.0, segments[0].get("start", 0.0))
    new = []
    for seg, tf in zip(segments, tts_files):
        dur = get_media_duration(tf)
        if dur <= 0:
            dur = max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
        start = t
        end = start + dur
        new.append({"start": start, "end": end, "text": seg.get("text","")})
        t = end
    total = new[-1]["end"] if new else 0.0
    return new, total

def create_dubbed_audio(segments, tts_files, output_audio, total_duration):
    sample_rate = 22050
    base_audio = np.zeros(int(max(total_duration, 0.0) * sample_rate) + 1, dtype=np.float32)
    for seg, tts_file in zip(segments, tts_files):
        if not os.path.exists(tts_file):
            continue
        try:
            audio, sr = librosa.load(tts_file, sr=sample_rate)
            start_sample = int(seg["start"] * sample_rate)
            end_sample = start_sample + len(audio)
            if end_sample > len(base_audio):
                pad = end_sample - len(base_audio)
                base_audio = np.pad(base_audio, (0, pad), constant_values=0.0)
            base_audio[start_sample:end_sample] += audio
        except:
            continue
    peak = np.max(np.abs(base_audio)) if base_audio.size else 0.0
    if peak > 1.0:
        base_audio = base_audio / peak
    sf.write(output_audio, base_audio, sample_rate)
    return True

def slow_video_to_duration(video_path: str, target_duration: float, output_path: str, preset: str = "ultrafast", crf: str = "23"):
    orig = get_video_duration(video_path)
    if orig <= 0 or target_duration <= 0:
        return False
    ratio = target_duration / orig
    if ratio <= 1.001:
        try:
            shutil.copy(video_path, output_path)
            return True
        except:
            return False
    cmd = [
        'ffmpeg', '-i', video_path,
        '-an',
        '-filter:v', f'setpts={ratio}*PTS',
        '-c:v', 'libx264', '-preset', preset, '-crf', crf,
        '-y', output_path
    ]
    return run_ffmpeg(cmd)

def merge_video_audio_ffmpeg(video_path, audio_path, output_path, copy_video=False):
    if copy_video:
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest', '-y', output_path
        ]
    else:
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest', '-y', output_path
        ]
    return run_ffmpeg(cmd)

def save_transcript(segs, orig, hindi, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("HINGLISH DUBBING TRANSCRIPT\n" + "="*80 + "\n\n")
            for i, (s, o, h) in enumerate(zip(segs, orig, hindi)):
                f.write(f"[{s['start']:.1f}s - {s['end']:.1f}s]\n")
                f.write(f"EN: {o}\n")
                f.write(f"HI: {h}\n\n")
        return True
    except:
        return False

# ── MAIN PROCESS (with timers) ───────────────────────────────
def process_video(input_path, output_path):
    print(f"Processing: {input_path}")
    T0 = time.perf_counter()
    timings = {}
    try:
        video_duration = get_video_duration(input_path)
        print(f"Duration: {video_duration:.1f}s")

        with tempfile.TemporaryDirectory() as tmp:
            # Extract audio
            t_s = time.perf_counter()
            audio_file = os.path.join(tmp, "audio.wav")
            print("Extracting audio...")
            extract_audio_ffmpeg(input_path, audio_file)
            timings["extract"] = time.perf_counter() - t_s

            # Transcribe
            t_s = time.perf_counter()
            print("Transcribing...")
            segs = transcribe_audio(audio_file)
            if not segs:
                raise ValueError("No speech")
            valid = []
            for s in segs:
                start = max(0.0, float(s.get("start", 0.0)))
                end = min(video_duration, float(s.get("end", 0.0)))
                text = str(s.get("text", "")).strip()
                if text and end > start:
                    valid.append({"start": start, "end": end, "text": text})
            print(f"Found {len(valid)} segments")
            timings["asr"] = time.perf_counter() - t_s

            # Translate
            t_s = time.perf_counter()
            print("Translating to Hinglish...")
            orig_texts = [s["text"] for s in valid]
            hindi = translate_to_hinglish_sync(orig_texts)
            print("Translation done.")
            timings["translate"] = time.perf_counter() - t_s

            # TTS
            t_s = time.perf_counter()
            print("Generating speech...")
            voice = get_voice()
            tts_files = [os.path.join(tmp, f"s_{i:03d}.mp3") for i in range(len(valid))]
            asyncio.run(batch_tts(hindi, tts_files, voice))
            timings["tts"] = time.perf_counter() - t_s

            # Reschedule by TTS durations
            t_s = time.perf_counter()
            print("Rescheduling segments to TTS durations...")
            sched_segments, total_dubbed_dur = reschedule_segments_by_tts(valid, tts_files)
            # Save transcript
            txt_path = output_path.replace('.mp4', '_transcript.txt')
            save_transcript(sched_segments, orig_texts, hindi, txt_path)
            timings["reschedule+transcript"] = time.perf_counter() - t_s

            # Create dubbed audio
            t_s = time.perf_counter()
            print("Creating audio track...")
            final_audio = os.path.join(tmp, "dubbed.wav")
            create_dubbed_audio(sched_segments, tts_files, final_audio, total_dubbed_dur)
            audio_out = output_path.replace('.mp4', '_audio.wav')
            shutil.copy(final_audio, audio_out)
            timings["mix_audio"] = time.perf_counter() - t_s

            # Slow video if needed
            t_s = time.perf_counter()
            print("Slowing video to match dubbed audio (if needed)...")
            video_for_merge = input_path
            slowed_video = os.path.join(tmp, "video_slow.mp4")
            slowed_ok = slow_video_to_duration(input_path, total_dubbed_dur, slowed_video)
            if slowed_ok and get_video_duration(slowed_video) > video_duration + 0.5:
                video_for_merge = slowed_video
            timings["slow_video"] = time.perf_counter() - t_s

            # Merge
            t_s = time.perf_counter()
            print("Merging video and audio...")
            copy_video = (video_for_merge == input_path)  # no slow-down => stream copy
            merge_video_audio_ffmpeg(video_for_merge, final_audio, output_path, copy_video=copy_video)
            timings["merge"] = time.perf_counter() - t_s

        total = time.perf_counter() - T0
        print("\n=== Timing Summary (seconds) ===")
        for k in ["extract", "asr", "translate", "tts", "reschedule+transcript", "mix_audio", "slow_video", "merge"]:
            v = timings.get(k, 0.0)
            print(f"{k:>20}: {v:.2f}")
        print(f"{'TOTAL':>20}: {total:.2f}\n")
        return True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

# ── CHUNKED UPLOAD ENDPOINTS ────────────────────────────────
def is_allowed_ext(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTS

@app.post("/upload/init")
def upload_init():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    size = int(data.get("size") or 0)
    if not filename or not is_allowed_ext(filename):
        return jsonify({"error": "Invalid or unsupported file"}), 400
    upload_id = uuid.uuid4().hex
    session_dir = Path(TMP_DIR, upload_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = {"filename": secure_filename(filename), "size": size, "created": time.time()}
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return jsonify({"upload_id": upload_id})

@app.post("/upload/chunk")
def upload_chunk():
    upload_id = request.form.get("upload_id", "")
    index = request.form.get("index", "")
    total = request.form.get("total", "")
    file = request.files.get("chunk")

    if not upload_id or index == "" or not total or file is None:
        return jsonify({"error": "Missing fields"}), 400

    try:
        idx = int(index)
        _ = int(total)
    except:
        return jsonify({"error": "Bad index/total"}), 400

    session_dir = Path(TMP_DIR, upload_id)
    if not session_dir.exists():
        return jsonify({"error": "Invalid upload_id"}), 400

    chunk_path = session_dir / f"chunk_{idx:06d}"
    try:
        file.save(chunk_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "received": idx})

@app.post("/upload/complete")
def upload_complete():
    data = request.get_json(silent=True) or {}
    upload_id = data.get("upload_id", "")
    if not upload_id:
        return jsonify({"error": "upload_id required"}), 400

    session_dir = Path(TMP_DIR, upload_id)
    meta_path = session_dir / "meta.json"
    if not session_dir.exists() or not meta_path.exists():
        return jsonify({"error": "Invalid session"}), 400

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    filename = meta.get("filename", "")
    if not filename or not is_allowed_ext(filename):
        return jsonify({"error": "Unsupported file"}), 400

    ext = filename.rsplit(".", 1)[1].lower()
    fid = upload_id[:12]
    final_inp = Path(UPLOAD_DIR, f"{fid}.{ext}")

    chunk_files = sorted(session_dir.glob("chunk_*"))
    if not chunk_files:
        return jsonify({"error": "No chunks received"}), 400

    with open(final_inp, "wb") as out_f:
        for ch in chunk_files:
            with open(ch, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=1024 * 1024)

    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except:
        pass

    out_path = Path(OUTPUT_DIR, f"{fid}.mp4")
    success = process_video(str(final_inp), str(out_path))

    try:
        os.remove(final_inp)
    except:
        pass

    if success:
        return jsonify({"success": True, "id": fid})
    else:
        return jsonify({"error": "Processing failed"}), 500

# Legacy direct upload
@app.post("/upload")
def legacy_upload():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video"}), 400
        vf = request.files["video"]
        if not vf.filename:
            return jsonify({"error": "No file"}), 400

        ext = vf.filename.lower().rsplit(".", 1)[1] if "." in vf.filename else ""
        if ext not in ALLOWED_EXTS:
            return jsonify({"error": f"Unsupported: {ext}"}), 400

        fid = uuid.uuid4().hex[:12]
        inp = os.path.join(UPLOAD_DIR, f"{fid}.{ext}")
        out = os.path.join(OUTPUT_DIR, f"{fid}.mp4")

        vf.save(inp)
        success = process_video(inp, out)

        try:
            os.remove(inp)
        except:
            pass

        if success:
            return jsonify({"success": True, "id": fid})
        else:
            return jsonify({"error": "Processing failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── UI ───────────────────────────────────────────────────────
INDEX_HTML = '''<!DOCTYPE html>
<html>
<head><title>Hinglish Dubber</title>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Arial;max-width:700px;margin:40px auto;padding:20px;background:#f5f5f5}
.box{background:#fff;padding:24px;border-radius:10px}
h1{text-align:center;color:#333;margin-bottom:18px}
.alert{background:#fff3cd;padding:12px;border-radius:6px;margin:12px 0;color:#856404}
input,button{display:block;width:100%;margin:10px 0;padding:12px;border-radius:6px;border:1px solid #ddd}
button{background:#007bff;color:#fff;border:none;cursor:pointer}
button:hover{background:#0056b3}
.status{display:none;padding:12px;background:#d1ecf1;border-radius:6px;margin:14px 0}
.progress{height:10px;background:#eee;border-radius:5px;overflow:hidden;margin-top:8px}
.bar{height:10px;background:#28a745;width:0%}
.small{font-size:12px;color:#666;margin-top:6px}
</style>
</head>
<body>
<div class="box">
  <h1>🎤 Hinglish Dubber</h1>
  <div class="alert">
    <strong>✅ Chunked Uploads + Timed Pipeline</strong><br>
    Large videos upload in chunks; stage-wise timing + total time printed
  </div>

  <form id="form">
    <input type="file" name="video" id="video" accept="video/*" required>
    <button type="submit" id="btn">Convert</button>
  </form>

  <div class="status" id="status">Preparing...</div>
  <div class="progress"><div class="bar" id="bar"></div></div>
  <div class="small" id="info"></div>
</div>

<script>
const CHUNK_SIZE = 8 * 1024 * 1024; // 8MB

async function uploadInChunks(file) {
  const initResp = await fetch('/upload/init', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ filename: file.name, size: file.size })
  });
  if (!initResp.ok) throw new Error('Init failed');
  const { upload_id } = await initResp.json();

  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  let uploaded = 0;

  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end = Math.min(start + CHUNK_SIZE, file.size);
    const blob = file.slice(start, end);

    const fd = new FormData();
    fd.append('upload_id', upload_id);
    fd.append('index', String(i));
    fd.append('total', String(totalChunks));
    fd.append('chunk', new File([blob], `chunk_${i}`, { type: file.type || 'application/octet-stream' }));

    const r = await fetch('/upload/chunk', { method: 'POST', body: fd });
    if (!r.ok) throw new Error('Chunk upload failed: ' + i);

    uploaded = end;
    const pct = Math.floor((uploaded / file.size) * 100);
    document.getElementById('bar').style.width = pct + '%';
    document.getElementById('info').textContent = `Uploaded ${i+1}/${totalChunks} chunks (${pct}%)`;
  }

  const comp = await fetch('/upload/complete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ upload_id })
  });
  if (!comp.ok) throw new Error('Complete failed');
  return comp.json();
}

document.getElementById('form').onsubmit = async (e) => {
  e.preventDefault();
  const file = document.getElementById('video').files[0];
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  const bar = document.getElementById('bar');
  const info = document.getElementById('info');

  if (!file) return;

  btn.disabled = true;
  btn.textContent = 'Uploading...';
  status.style.display = 'block';
  status.textContent = 'Uploading in chunks...';
  bar.style.width = '0%';
  info.textContent = '';

  try {
    const out = await uploadInChunks(file);
    if (out && out.success) {
      status.textContent = 'Processing... (check terminal for timing)';
      window.location.href = `/result?id=${out.id}`;
    } else {
      throw new Error(out.error || 'Unknown error');
    }
  } catch (err) {
    alert('Error: ' + err.message);
    status.style.display = 'none';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Convert';
  }
};
</script>
</body>
</html>
'''

@app.get("/")
def home():
    return INDEX_HTML

@app.get("/result")
def result():
    fid = request.args.get('id', '')
    if not fid:
        return "Invalid", 400
    return f'''<!DOCTYPE html>
<html><head><title>Done</title>
<style>
body{{font-family:Arial;max-width:800px;margin:50px auto;padding:20px;background:#f5f5f5}}
.box{{background:#fff;padding:30px;border-radius:10px;text-align:center}}
video{{max-width:100%;margin:20px 0;border-radius:10px}}
a{{display:inline-block;margin:10px;padding:12px 24px;background:#007bff;color:#fff;text-decoration:none;border-radius:5px}}
a:hover{{background:#0056b3}}
.success{{background:#d4edda;padding:15px;border-radius:5px;margin:20px 0;color:#155724}}
</style></head><body>
<div class="box">
<h1>✅ Done!</h1>
<div class="success">Stage-wise timing printed in server logs. Video auto-extended to match HI speech.</div>
<video src="/download/{fid}.mp4" controls></video>
<div>
<a href="/download/{fid}.mp4" download>Video</a>
<a href="/download/{fid}_audio.wav" download>Audio</a>
<a href="/download/{fid}_transcript.txt" download>Transcript</a>
<a href="/">Another</a>
</div>
</div></body></html>'''

@app.get("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except:
        abort(404)

if __name__ == "__main__":
    print("="*60)
    print("Hinglish Dubber - Chunked Uploads + No-Skip Audio + Video Slowdown + Timed Pipeline")
    print("="*60)
    if not CEREBRAS_API_KEY:
        print("⚠️  Set CEREBRAS_API_KEY in .env!")
    print("Tip: set USE_FASTER_WHISPER=1 to enable faster-whisper if installed")
    print("Server: http://0.0.0.0:7860")
    print("="*60)
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
