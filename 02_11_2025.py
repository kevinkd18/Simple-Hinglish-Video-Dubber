#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# COMPLETE FIXED VERSION:
# - Background job processing for long videos (2+ hours)
# - No HTTP timeouts - immediate response with job ID
# - Real-time progress polling
# - All previous fixes included

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
from threading import Thread
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
import whisper
import edge_tts

from flask import Flask, request, send_from_directory, abort, jsonify
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
try:
    from cerebras.cloud.sdk import RateLimitError
except Exception:
    class RateLimitError(Exception):
        pass
from werkzeug.utils import secure_filename
from pathlib import Path
import shutil

# Optional faster-whisper
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
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
FAST_WHISPER_MODEL = os.environ.get("FAST_WHISPER_MODEL", "base")
USE_FASTER_WHISPER = os.environ.get("USE_FASTER_WHISPER", "0") == "1"
TTS_CONCURRENCY = int(os.environ.get("TTS_CONCURRENCY", "10"))
ALLOWED_EXTS = {"mp4", "mov", "mkv", "webm", "avi"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hinglish-dubber")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# ── JOB MANAGEMENT ──────────────────────────────────────────
jobs = defaultdict(dict)

def cleanup_old_jobs():
    """Remove jobs older than 24 hours"""
    current_time = time.time()
    expired_jobs = []
    for job_id, job in jobs.items():
        if current_time - job.get('created_at', 0) > 24 * 3600:
            expired_jobs.append(job_id)
    
    for job_id in expired_jobs:
        try:
            # Cleanup temporary files
            input_path = jobs[job_id].get('input_path')
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
            del jobs[job_id]
        except:
            pass

# ── TTS VOICE PICKER ─────────────────────────────────────────
def get_voice() -> str:
    return os.environ.get("VOICE_NAME", "en-IN-NeerjaNeural")

# ── WHISPER LAZY-LOAD ──────────────────────────────────────
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

# ── TTS GENERATION ──────────────────────────────────────────
async def gen_speech(text: str, voice: str, output: str):
    if not text or not text.strip():
        return False
    try:
        clean = re.sub(r'[^\w\s.,!?-]', '', text).strip()
        comm = edge_tts.Communicate(text=clean, voice=voice)
        await comm.save(output)
        return os.path.exists(output) and os.path.getsize(output) > 1000
    except:
        return False

async def batch_tts(texts: List[str], files: List[str], voice: str):
    sem = asyncio.Semaphore(TTS_CONCURRENCY)
    
    async def gen_one(txt, f):
        async with sem:
            success = await gen_speech(txt, voice, f)
            return success
    
    tasks = [gen_one(t, f) for t, f in zip(texts, files) if t and t.strip()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if r is True)
    logger.info(f"TTS completed: {success_count}/{len(texts)}")
    return results

# ── AUDIO ASSEMBLY / VIDEO SLOWDOWN / MERGE ─────────────────
def reschedule_segments_by_tts(segments: List[Dict], tts_files: List[str]) -> (List[Dict], float):
    """
    Preserve original timing relationships while adjusting for TTS durations
    """
    if not segments:
        return [], 0.0
    
    new_segments = []
    current_time = segments[0]["start"]
    
    for i, (seg, tts_file) in enumerate(zip(segments, tts_files)):
        tts_duration = get_media_duration(tts_file)
        if tts_duration <= 0:
            tts_duration = seg["end"] - seg["start"]
        
        original_gap = 0.0
        if i < len(segments) - 1:
            original_gap = segments[i+1]["start"] - seg["end"]
        
        start_time = current_time
        end_time = start_time + tts_duration
        
        new_segments.append({
            "start": start_time,
            "end": end_time, 
            "text": seg["text"]
        })
        
        current_time = end_time + original_gap
    
    total_duration = new_segments[-1]["end"] if new_segments else 0.0
    return new_segments, total_duration

def create_dubbed_audio(segments, tts_files, output_audio, total_duration):
    sample_rate = 22050
    total_samples = int(total_duration * sample_rate) + 1
    base_audio = np.zeros(total_samples, dtype=np.float32)
    
    for seg, tts_file in zip(segments, tts_files):
        if not os.path.exists(tts_file):
            continue
        try:
            audio, sr = librosa.load(tts_file, sr=sample_rate)
            start_sample = int(seg["start"] * sample_rate)
            end_sample = start_sample + len(audio)
            
            if end_sample > len(base_audio):
                extension = end_sample - len(base_audio)
                base_audio = np.pad(base_audio, (0, extension), mode='constant')
            
            segment_to_add = audio[:len(base_audio) - start_sample]
            base_audio[start_sample:start_sample + len(segment_to_add)] += segment_to_add
            
        except Exception as e:
            logger.error(f"Error mixing audio segment {tts_file}: {e}")
            continue
    
    peak = np.max(np.abs(base_audio)) if len(base_audio) > 0 else 1.0
    if peak > 1.0:
        base_audio = base_audio / peak
    
    sf.write(output_audio, base_audio, sample_rate)
    return True

def slow_video_to_duration(video_path: str, target_duration: float, output_path: str):
    orig_duration = get_video_duration(video_path)
    if orig_duration <= 0 or target_duration <= 0:
        return False
    
    if target_duration <= orig_duration:
        try:
            shutil.copy(video_path, output_path)
            return True
        except:
            return False
    
    ratio = target_duration / orig_duration
    
    cmd = [
        'ffmpeg', '-i', video_path, '-an',
        '-filter:v', f'setpts={ratio}*PTS',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
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
            '-c:v', 'libx264', '-preset', 'medium',
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

# ── BACKGROUND PROCESSING ───────────────────────────────────
def process_video_background(job_id: str):
    """Background processing for long videos"""
    job = jobs[job_id]
    try:
        input_path = job['input_path']
        output_path = job['output_path']
        fid = job['fid']
        
        # Step 1: Extract audio
        job['status'] = 'processing'
        job['progress'] = 10
        job['message'] = 'Extracting audio from video...'
        
        with tempfile.TemporaryDirectory() as tmp:
            audio_file = os.path.join(tmp, "audio.wav")
            if not extract_audio_ffmpeg(input_path, audio_file):
                raise ValueError("Audio extraction failed")
            
            # Step 2: Transcribe
            job['progress'] = 20
            job['message'] = 'Transcribing audio...'
            segs = transcribe_audio(audio_file)
            if not segs:
                raise ValueError("No speech detected")
                
            valid = []
            video_duration = get_video_duration(input_path)
            for s in segs:
                start = max(0.0, float(s.get("start", 0.0)))
                end = min(video_duration, float(s.get("end", 0.0)))
                text = str(s.get("text", "")).strip()
                if text and end > start:
                    valid.append({"start": start, "end": end, "text": text})
            
            job['progress'] = 30
            job['message'] = f'Found {len(valid)} speech segments'
            
            # Step 3: Translate
            job['progress'] = 40
            job['message'] = 'Translating to Hinglish...'
            orig_texts = [s["text"] for s in valid]
            hindi = translate_to_hinglish_sync(orig_texts)
            
            # Step 4: TTS
            job['progress'] = 50
            job['message'] = 'Generating Hindi speech...'
            voice = get_voice()
            tts_files = [os.path.join(tmp, f"s_{i:03d}.mp3") for i in range(len(valid))]
            asyncio.run(batch_tts(hindi, tts_files, voice))
            
            # Step 5: Reschedule segments
            job['progress'] = 70
            job['message'] = 'Synchronizing audio with video...'
            sched_segments, total_dubbed_dur = reschedule_segments_by_tts(valid, tts_files)
            
            # Save transcript
            txt_path = output_path.replace('.mp4', '_transcript.txt')
            save_transcript(sched_segments, orig_texts, hindi, txt_path)
            
            # Step 6: Create dubbed audio
            job['progress'] = 80
            job['message'] = 'Creating final audio track...'
            final_audio = os.path.join(tmp, "dubbed.wav")
            create_dubbed_audio(sched_segments, tts_files, final_audio, total_dubbed_dur)
            audio_out = output_path.replace('.mp4', '_audio.wav')
            shutil.copy(final_audio, audio_out)
            
            # Step 7: Slow video if needed
            job['progress'] = 90
            job['message'] = 'Finalizing video...'
            video_for_merge = input_path
            slowed_video = os.path.join(tmp, "video_slow.mp4")
            slowed_ok = slow_video_to_duration(input_path, total_dubbed_dur, slowed_video)
            if slowed_ok and get_video_duration(slowed_video) > video_duration + 0.5:
                video_for_merge = slowed_video
            
            # Step 8: Merge
            copy_video = (video_for_merge == input_path)
            merge_video_audio_ffmpeg(video_for_merge, final_audio, output_path, copy_video=copy_video)
        
        # Cleanup input file
        try:
            os.remove(input_path)
        except:
            pass
            
        job['status'] = 'completed'
        job['progress'] = 100
        job['message'] = 'Processing completed successfully!'
        
    except Exception as e:
        job['status'] = 'failed'
        job['message'] = f'Error: {str(e)}'
        logger.error(f"Job {job_id} failed: {e}")
        traceback.print_exc()

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
    
    # Create background job instead of direct processing
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Waiting to start processing...',
        'fid': fid,
        'input_path': str(final_inp),
        'output_path': str(out_path),
        'created_at': time.time()
    }
    
    # Start background processing
    thread = Thread(target=process_video_background, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True, "job_id": job_id})

@app.get("/job/status/<job_id>")
def job_status(job_id):
    cleanup_old_jobs()
    
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "status": job['status'],
        "progress": job['progress'],
        "message": job['message'],
        "fid": job.get('fid'),
        "created_at": job.get('created_at')
    })

# Legacy direct upload (for small videos)
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
        
        # For small files, process directly
        success = False
        try:
            success = process_video_background_direct(inp, out, fid)
        except Exception as e:
            logger.error(f"Direct processing failed: {e}")

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

def process_video_background_direct(input_path, output_path, fid):
    """Direct processing for small videos (under 10 minutes)"""
    try:
        # Same logic as background processing but synchronous
        with tempfile.TemporaryDirectory() as tmp:
            audio_file = os.path.join(tmp, "audio.wav")
            if not extract_audio_ffmpeg(input_path, audio_file):
                return False
                
            segs = transcribe_audio(audio_file)
            if not segs:
                return False
                
            valid = []
            video_duration = get_video_duration(input_path)
            for s in segs:
                start = max(0.0, float(s.get("start", 0.0)))
                end = min(video_duration, float(s.get("end", 0.0)))
                text = str(s.get("text", "")).strip()
                if text and end > start:
                    valid.append({"start": start, "end": end, "text": text})
            
            orig_texts = [s["text"] for s in valid]
            hindi = translate_to_hinglish_sync(orig_texts)
            
            voice = get_voice()
            tts_files = [os.path.join(tmp, f"s_{i:03d}.mp3") for i in range(len(valid))]
            asyncio.run(batch_tts(hindi, tts_files, voice))
            
            sched_segments, total_dubbed_dur = reschedule_segments_by_tts(valid, tts_files)
            
            txt_path = output_path.replace('.mp4', '_transcript.txt')
            save_transcript(sched_segments, orig_texts, hindi, txt_path)
            
            final_audio = os.path.join(tmp, "dubbed.wav")
            create_dubbed_audio(sched_segments, tts_files, final_audio, total_dubbed_dur)
            audio_out = output_path.replace('.mp4', '_audio.wav')
            shutil.copy(final_audio, audio_out)
            
            video_for_merge = input_path
            slowed_video = os.path.join(tmp, "video_slow.mp4")
            slowed_ok = slow_video_to_duration(input_path, total_dubbed_dur, slowed_video)
            if slowed_ok and get_video_duration(slowed_video) > video_duration + 0.5:
                video_for_merge = slowed_video
            
            copy_video = (video_for_merge == input_path)
            merge_video_audio_ffmpeg(video_for_merge, final_audio, output_path, copy_video=copy_video)
            
        return True
        
    except Exception as e:
        logger.error(f"Direct processing error: {e}")
        return False

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
button:disabled{background:#6c757d;cursor:not-allowed}
.status{display:none;padding:12px;background:#d1ecf1;border-radius:6px;margin:14px 0}
.progress{height:10px;background:#eee;border-radius:5px;overflow:hidden;margin-top:8px}
.bar{height:10px;background:#28a745;width:0%;transition:width 0.3s}
.small{font-size:12px;color:#666;margin-top:6px}
.job-info{padding:10px;background:#f8f9fa;border-radius:5px;margin:10px 0}
</style>
</head>
<body>
<div class="box">
  <h1>🎤 Hinglish Dubber</h1>
  <div class="alert">
    <strong>✅ Background Processing Enabled</strong><br>
    Large videos (2+ hours) now process in background without timeouts
  </div>

  <form id="form">
    <input type="file" name="video" id="video" accept="video/*" required>
    <button type="submit" id="btn">Convert to Hinglish</button>
  </form>

  <div class="status" id="status">Preparing upload...</div>
  <div class="progress"><div class="bar" id="bar"></div></div>
  <div class="small" id="info"></div>
  <div class="job-info" id="jobInfo" style="display:none"></div>
</div>

<script>
const CHUNK_SIZE = 8 * 1024 * 1024;

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
    fd.append('chunk', new File([blob], `chunk_${i}`, { type: file.type }));

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

async function pollJobStatus(jobId) {
  const statusEl = document.getElementById('status');
  const bar = document.getElementById('bar');
  const infoEl = document.getElementById('info');
  const jobInfoEl = document.getElementById('jobInfo');
  
  jobInfoEl.style.display = 'block';
  
  return new Promise((resolve, reject) => {
    const checkInterval = setInterval(async () => {
      try {
        const response = await fetch(`/job/status/${jobId}`);
        const data = await response.json();
        
        statusEl.textContent = data.message;
        bar.style.width = data.progress + '%';
        infoEl.textContent = `Progress: ${data.progress}%`;
        jobInfoEl.innerHTML = `<strong>Job Status:</strong> ${data.status}<br>
                              <strong>Message:</strong> ${data.message}<br>
                              <strong>Progress:</strong> ${data.progress}%`;
        
        if (data.status === 'completed') {
          clearInterval(checkInterval);
          resolve({ success: true, id: data.fid });
        } else if (data.status === 'failed') {
          clearInterval(checkInterval);
          reject(new Error(data.message));
        }
      } catch (error) {
        clearInterval(checkInterval);
        reject(error);
      }
    }, 3000);
  });
}

document.getElementById('form').onsubmit = async (e) => {
  e.preventDefault();
  const file = document.getElementById('video').files[0];
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  const bar = document.getElementById('bar');
  const info = document.getElementById('info');
  const jobInfo = document.getElementById('jobInfo');

  if (!file) return;

  btn.disabled = true;
  btn.textContent = 'Uploading...';
  status.style.display = 'block';
  status.textContent = 'Uploading in chunks...';
  bar.style.width = '0%';
  info.textContent = '';
  jobInfo.style.display = 'none';

  try {
    const result = await uploadInChunks(file);
    
    if (result.success && result.job_id) {
      status.textContent = 'Processing started...';
      info.textContent = 'Video is being processed in background...';
      
      const finalResult = await pollJobStatus(result.job_id);
      
      if (finalResult.success) {
        status.textContent = 'Processing complete! Redirecting...';
        setTimeout(() => {
          window.location.href = `/result?id=${finalResult.id}&job_id=${result.job_id}`;
        }, 2000);
      }
    } else {
      throw new Error(result.error || 'Unknown error');
    }
  } catch (err) {
    alert('Error: ' + err.message);
    status.style.display = 'none';
    jobInfo.style.display = 'none';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Convert to Hinglish';
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
    job_id = request.args.get('job_id', '')
    
    if job_id:
        job = jobs.get(job_id)
        if job and job['status'] == 'completed':
            fid = job['fid']
    
    if not fid:
        return "Invalid video ID", 400
        
    return f'''<!DOCTYPE html>
<html>
<head>
<title>Processing Complete</title>
<meta charset="utf-8"/>
<style>
body{{font-family:Arial;max-width:800px;margin:50px auto;padding:20px;background:#f5f5f5}}
.box{{background:#fff;padding:30px;border-radius:10px;text-align:center}}
video{{max-width:100%;margin:20px 0;border-radius:10px}}
a{{display:inline-block;margin:10px;padding:12px 24px;background:#007bff;color:#fff;text-decoration:none;border-radius:5px}}
a:hover{{background:#0056b3}}
.success{{background:#d4edda;padding:15px;border-radius:5px;margin:20px 0;color:#155724}}
.downloads{{margin:20px 0}}
</style>
</head>
<body>
<div class="box">
<h1>✅ Hinglish Dubbing Complete!</h1>
<div class="success">
    Your video has been successfully processed with Hinglish audio dubbing.
</div>
<video src="/download/{fid}.mp4" controls poster="/download/{fid}.jpg"></video>
<div class="downloads">
    <a href="/download/{fid}.mp4" download>Download Video</a>
    <a href="/download/{fid}_audio.wav" download>Download Audio</a>
    <a href="/download/{fid}_transcript.txt" download>Download Transcript</a>
    <a href="/">Process Another Video</a>
</div>
<p style="color:#666;margin-top:20px;font-size:14px">
    The original English audio has been replaced with Hinglish speech while preserving video quality.
</p>
</div>
</body>
</html>'''

@app.get("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except:
        abort(404)

if __name__ == "__main__":
    print("="*60)
    print("Hinglish Dubber - Background Processing Enabled")
    print("Supports 2+ hour videos without timeouts")
    print("="*60)
    if not CEREBRAS_API_KEY:
        print("⚠️  Set CEREBRAS_API_KEY in .env!")
    print("Server: http://0.0.0.0:7860")
    print("="*60)
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
