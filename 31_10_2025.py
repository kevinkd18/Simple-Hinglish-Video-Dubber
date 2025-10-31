#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# COMPLETE FIX: No change to TTS/content, prevent speech skipping by rescheduling
# segments to TTS durations and slowing video uniformly to match final audio.

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
from typing import List, Dict

import numpy as np
import soundfile as sf
import librosa
import whisper
import edge_tts

from flask import Flask, request, send_from_directory, abort, jsonify
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

# ── CONFIG ───────────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore")

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    print("⚠️  CEREBRAS_API_KEY missing! Translation will fail!")

client = Cerebras(api_key=CEREBRAS_API_KEY) if CEREBRAS_API_KEY else None

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
WHISPER_MODEL = "base"
TTS_CONCURRENCY = 10

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1500 * 1024 * 1024

# ── CORE ─────────────────────────────────────────────────────

print("Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("Ready!")

def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def get_media_duration(path: str) -> float:
    # Works for audio/video; returns seconds (float)
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

def translate_to_hinglish_sync(texts: List[str]) -> List[str]:
    if not texts or not client:
        print("⚠️  No client or texts")
        return texts
    results = []
    for i in range(0, len(texts), 5):
        batch = texts[i:i+5]
        try:
            text_input = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
            prompt = f"""Convert these English sentences to Hinglish (mix of Hindi and English in Roman script).
Make it natural like Indians speak. Examples:
- "Hello everyone" → "Namaste sabhi ko"
- "Thank you" → "Dhanyavaad" 
- "Let's start" → "Chalo shuru karte hain"

Convert these sentences. Return ONLY a JSON array like ["hinglish1", "hinglish2"]:

{text_input}"""
            response = client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[
                    {"role": "system", "content": "You convert English to Hinglish. Always return valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=2000
            )
            content = response.choices[0].message.content
            m = re.search(r'\[.*?\]', content, re.DOTALL)
            if m:
                arr = json.loads(m.group(0))
                results.extend(arr if len(arr)==len(batch) else batch)
            else:
                results.extend(batch)
        except Exception:
            traceback.print_exc()
            results.extend(batch)
        time.sleep(0.5)
    if len(results) != len(texts):
        return texts
    return results

def get_voice():
    return "en-IN-NeerjaNeural"

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

async def batch_tts(texts, files, voice):
    sem = asyncio.Semaphore(TTS_CONCURRENCY)
    async def gen_one(txt, f):
        async with sem:
            return await gen_speech(txt, voice, f)
    tasks = [gen_one(t, f) for t, f in zip(texts, files) if t and t.strip()]
    return await asyncio.gather(*tasks, return_exceptions=True)

def reschedule_segments_by_tts(segments: List[Dict], tts_files: List[str]) -> (List[Dict], float):
    """
    Make segments sequential by TTS durations:
      start_0' = original start_0
      end_i' = start_i' + dur(tts_i)
      start_{i+1}' = end_i'
    """
    if not segments:
        return [], 0.0
    # Base start: keep original first start for minimal shift
    t = max(0.0, segments[0].get("start", 0.0))
    new = []
    for seg, tf in zip(segments, tts_files):
        dur = get_media_duration(tf)
        if dur <= 0:
            # Fallback to original segment length if TTS file missing
            dur = max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
        start = t
        end = start + dur
        new.append({"start": start, "end": end, "text": seg.get("text","")})
        t = end
    total = new[-1]["end"] if new else 0.0
    return new, total

def create_dubbed_audio(segments, tts_files, output_audio, total_duration):
    sample_rate = 22050
    base_audio = np.zeros(int(max(total_duration, 0.0) * sample_rate)+1, dtype=np.float32)
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
            # Mix (accumulate) instead of overwrite to avoid cutting previous tails
            base_audio[start_sample:end_sample] += audio
        except:
            continue
    # Prevent clipping
    peak = np.max(np.abs(base_audio)) if base_audio.size else 0.0
    if peak > 1.0:
        base_audio = base_audio / peak
    sf.write(output_audio, base_audio, sample_rate)
    return True

def slow_video_to_duration(video_path: str, target_duration: float, output_path: str, preset: str = "ultrafast"):
    orig = get_video_duration(video_path)
    if orig <= 0 or target_duration <= 0:
        return False
    ratio = target_duration / orig
    if ratio <= 1.001:
        try:
            import shutil
            shutil.copy(video_path, output_path)
            return True
        except:
            return False
    cmd = [
        'ffmpeg', '-i', video_path,
        '-an',
        '-filter:v', f'setpts={ratio}*PTS',
        '-c:v', 'libx264', '-preset', preset,
        '-y', output_path
    ]
    return run_ffmpeg(cmd)

def merge_video_audio_ffmpeg(video_path, audio_path, output_path):
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

def process_video(input_path, output_path):
    print(f"Processing: {input_path}")
    t_start = time.time()
    try:
        video_duration = get_video_duration(input_path)
        print(f"Duration: {video_duration:.1f}s")

        with tempfile.TemporaryDirectory() as tmp:
            # Extract audio
            audio_file = os.path.join(tmp, "audio.wav")
            print("Extracting audio...")
            extract_audio_ffmpeg(input_path, audio_file)

            # Transcribe
            print("Transcribing...")
            result = whisper_model.transcribe(audio_file, language='en')
            segs = result.get("segments", [])
            if not segs:
                raise ValueError("No speech")

            # Clean segments (original times)
            valid = []
            for s in segs:
                start = max(0, s.get("start", 0))
                end = min(video_duration, s.get("end", 0))
                text = s.get("text", "").strip()
                if text and end > start:
                    valid.append({"start": start, "end": end, "text": text})
            print(f"Found {len(valid)} segments")

            # Translate
            print("Translating to Hinglish...")
            orig_texts = [s["text"] for s in valid]
            hindi = translate_to_hinglish_sync(orig_texts)
            print("Translation done.")

            # TTS
            print("Generating speech...")
            voice = get_voice()
            tts_files = [os.path.join(tmp, f"s_{i:03d}.mp3") for i in range(len(valid))]
            asyncio.run(batch_tts(hindi, tts_files, voice))

            # Re-schedule segments to TTS lengths (prevents tail overwrite/skip)
            print("Rescheduling segments to TTS durations...")
            sched_segments, total_dubbed_dur = reschedule_segments_by_tts(valid, tts_files)

            # Save transcript with rescheduled times
            txt_path = output_path.replace('.mp4', '_transcript.txt')
            save_transcript(sched_segments, orig_texts, hindi, txt_path)

            # Create dubbed audio over new schedule
            print("Creating audio track...")
            final_audio = os.path.join(tmp, "dubbed.wav")
            create_dubbed_audio(sched_segments, tts_files, final_audio, total_dubbed_dur)

            # Save separate audio
            audio_out = output_path.replace('.mp4', '_audio.wav')
            import shutil
            shutil.copy(final_audio, audio_out)

            # Slow video uniformly to match final dubbed audio duration
            print("Slowing video to match dubbed audio...")
            video_for_merge = input_path
            slowed_video = os.path.join(tmp, "video_slow.mp4")
            if slow_video_to_duration(input_path, total_dubbed_dur, slowed_video):
                video_for_merge = slowed_video

            # Merge
            print("Merging video and audio...")
            merge_video_audio_ffmpeg(video_for_merge, final_audio, output_path)

        print(f"\nDone in {time.time()-t_start:.1f}s!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

# ── UI ───────────────────────────────────────────────────────

@app.route("/")
def home():
    return '''<!DOCTYPE html>
<html><head><title>Hinglish Dubber</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Arial;max-width:600px;margin:50px auto;padding:20px;background:#f5f5f5}
.box{background:#fff;padding:30px;border-radius:10px}
h1{text-align:center;color:#333;margin-bottom:20px}
.alert{background:#fff3cd;padding:15px;border-radius:5px;margin:15px 0;color:#856404}
input,button{display:block;width:100%;margin:10px 0;padding:12px;border-radius:5px;border:1px solid #ddd}
button{background:#007bff;color:#fff;border:none;cursor:pointer}
button:hover{background:#0056b3}
.status{display:none;padding:15px;background:#d1ecf1;border-radius:5px;margin:20px 0}
</style></head><body>
<div class="box">
<h1>🎤 Hinglish Dubber</h1>
<div class="alert">
<strong>✅ Translation Fixed + No-Skip Audio</strong><br>
Audio untouched; video auto-extends to fit full HI speech
</div>
<form id="form" enctype="multipart/form-data">
<input type="file" name="video" id="video" accept="video/*" required>
<button type="submit" id="btn">Convert</button>
</form>
<div class="status" id="status">Processing...</div>
</div>
<script>
document.getElementById('form').onsubmit = async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn');
    const status = document.getElementById('status');
    const formData = new FormData(e.target);
    btn.disabled = true;
    btn.textContent = 'Processing...';
    status.style.display = 'block';
    try {
        const resp = await fetch('/upload', {method: 'POST', body: formData});
        if (resp.ok) {
            const data = await resp.json();
            window.location.href = `/result?id=${data.id}`;
        } else {
            alert('Failed');
            btn.disabled = false;
            btn.textContent = 'Convert';
            status.style.display = 'none';
        }
    } catch (err) {
        alert('Error: ' + err.message);
        btn.disabled = false;
        btn.textContent = 'Convert';
        status.style.display = 'none';
    }
};
</script>
</body></html>'''

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video"}), 400
        vf = request.files["video"]
        if not vf.filename:
            return jsonify({"error": "No file"}), 400

        ext = vf.filename.lower().rsplit(".", 1)[1] if "." in vf.filename else ""
        if ext not in ["mp4", "mov", "mkv", "webm", "avi"]:
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

@app.route("/result")
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
<div class="success">HI speech fully preserved. Video extended to match audio.</div>
<video src="/download/{fid}.mp4" controls></video>
<div>
<a href="/download/{fid}.mp4" download>Video</a>
<a href="/download/{fid}_audio.wav" download>Audio</a>
<a href="/download/{fid}_transcript.txt" download>Transcript</a>
<a href="/">Another</a>
</div>
</div></body></html>'''

@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except:
        abort(404)

if __name__ == "__main__":
    print("="*60)
    print("Hinglish Dubber - No-Skip Audio + Video Slowdown")
    print("="*60)
    if not CEREBRAS_API_KEY:
        print("⚠️  Set CEREBRAS_API_KEY in .env!")
    print("Server: http://0.0.0.0:7860")
    print("="*60)
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
