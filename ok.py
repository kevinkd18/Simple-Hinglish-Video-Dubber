#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Simple Hinglish Video Dubber (Fixed for Librosa compatibility)

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
from typing import List, Tuple

import requests
import numpy as np
import soundfile as sf
import librosa
import whisper
import edge_tts

from flask import Flask, request, send_from_directory, abort
from dotenv import load_dotenv
from openai import OpenAI

# MoviePy imports
try:
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
    print("Using MoviePy 2.x import structure")
except ImportError:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        print("Using MoviePy 1.x import structure")
    except ImportError as e:
        print(f"MoviePy import failed: {e}")
        exit(1)

# ── CONFIGURATION ────────────────────────────────────────────
load_dotenv()
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")
warnings.filterwarnings("ignore", category=UserWarning)

# API Configuration
OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Directories and settings
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
WHISPER_MODEL = "base"
TTS_CONCURRENCY = 5
MAX_SEGMENT_LENGTH = 30

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── FLASK SETUP ──────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1500 * 1024 * 1024  # 1.5GB

# ── CORE FUNCTIONS ───────────────────────────────────────────

# Load Whisper model
print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded!")

def convert_to_hinglish(text_list):
    """Convert text to Hinglish with better error handling"""
    if not text_list:
        return []
    
    try:
        # Process in smaller batches for better reliability
        batch_size = 8
        results = []
        
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            
            prompt = """Convert these English sentences to natural conversational Hinglish (Roman script). 
Keep the meaning intact but make it sound like how Indians speak in mixed Hindi-English.
Return only a JSON array of converted sentences:

"""
            prompt += "\n".join(f"{j+1}. {text}" for j, text in enumerate(batch))
            
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            # Extract JSON array more robustly
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group(0))
                results.extend(batch_results)
            else:
                results.extend(batch)
        
        return results[:len(text_list)]  # Ensure same length
        
    except Exception as e:
        print(f"Hinglish conversion failed: {e}, using original text")
        return text_list

def detect_voice_gender(audio_file, start_time, end_time):
    """FIXED: Improved gender detection compatible with newer librosa versions"""
    try:
        duration = min(end_time - start_time, 3.0)
        y, sr = librosa.load(audio_file, sr=22050, offset=start_time, duration=duration)
        
        if len(y) < 1024:
            return "female"
        
        # FIXED: Removed deprecated 'threshold' parameter from pyin
        try:
            # Try newer librosa version syntax
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7')
            )
        except TypeError:
            # Fallback for older versions
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                threshold=0.1
            )
        
        # Use voiced_flag to filter out unvoiced parts
        if voiced_flag is not None and voiced_probs is not None:
            f0_clean = f0[voiced_flag & (voiced_probs > 0.5)]
        else:
            f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 5:
            avg_pitch = np.mean(f0_clean)
            return "female" if avg_pitch > 165 else "male"
        
        return "female"
        
    except Exception as e:
        print(f"Gender detection failed: {e}")
        return "female"

def get_voice_name(gender):
    """Better voice selection"""
    voices = {
        "male": "en-IN-PrabhatNeural",
        "female": "en-IN-NeerjaNeural"
    }
    return voices[gender]

async def generate_speech_with_retry(text, voice, output_file, max_retries=3):
    """TTS generation with retry logic"""
    for attempt in range(max_retries):
        try:
            # Clean text for better TTS
            clean_text = re.sub(r'[^\w\s.,!?-]', '', text)
            clean_text = clean_text.strip()
            
            if not clean_text:
                clean_text = "silence"
            
            comm = edge_tts.Communicate(text=clean_text, voice=voice)
            await comm.save(output_file)
            return True
            
        except Exception as e:
            print(f"TTS attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Create silent audio as fallback
                silent_audio = np.zeros(int(22050 * 2))
                sf.write(output_file.replace('.mp3', '.wav'), silent_audio, 22050)
                return False
            await asyncio.sleep(1)
    
    return False

def run_tts_batch(texts, voices, output_files):
    """Optimized batch TTS processing"""
    async def batch_generate():
        tasks = []
        semaphore = asyncio.Semaphore(TTS_CONCURRENCY)
        
        async def generate_one(text, voice, file):
            async with semaphore:
                return await generate_speech_with_retry(text, voice, file)
        
        for text, voice, file in zip(texts, voices, output_files):
            if text.strip():
                tasks.append(generate_one(text, voice, file))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    return asyncio.run(batch_generate())

def stretch_audio_to_duration(audio_data, sample_rate, target_duration):
    """Improved audio time stretching"""
    try:
        if len(audio_data) == 0:
            return np.zeros(int(target_duration * sample_rate))
        
        current_duration = len(audio_data) / sample_rate
        
        if abs(current_duration - target_duration) < 0.1:
            return audio_data
        
        stretch_rate = current_duration / target_duration
        stretch_rate = max(0.5, min(stretch_rate, 2.0))
        
        # FIXED: Use more compatible time_stretch method
        try:
            stretched = librosa.effects.time_stretch(audio_data, rate=stretch_rate)
        except:
            # Fallback method if time_stretch fails
            from scipy import signal
            stretched = signal.resample(audio_data, int(len(audio_data) / stretch_rate))
        
        target_samples = int(target_duration * sample_rate)
        
        if len(stretched) > target_samples:
            return stretched[:target_samples]
        elif len(stretched) < target_samples:
            padding = np.zeros(target_samples - len(stretched))
            return np.concatenate([stretched, padding])
        
        return stretched
        
    except Exception as e:
        print(f"Audio stretching failed: {e}")
        return np.zeros(int(target_duration * sample_rate))

def split_long_segments(segments, max_length=MAX_SEGMENT_LENGTH):
    """Split long segments for better processing"""
    result = []
    
    for segment in segments:
        duration = segment["end"] - segment["start"]
        
        if duration <= max_length:
            result.append(segment)
        else:
            # Split long segment
            text_parts = segment["text"].split('. ')
            if len(text_parts) <= 1:
                text_parts = segment["text"].split(' ')
            
            part_duration = duration / len(text_parts)
            
            for i, part in enumerate(text_parts):
                if part.strip():
                    result.append({
                        "start": segment["start"] + i * part_duration,
                        "end": segment["start"] + (i + 1) * part_duration,
                        "text": part.strip(),
                        "gender": segment["gender"]
                    })
    
    return result

def process_video_simple(input_path, output_path):
    """Optimized video processing with all compatibility fixes"""
    print(f"Processing video: {input_path}")
    
    video = None
    try:
        # Load video with error handling
        video = VideoFileClip(input_path)
        print(f"Video loaded: {video.duration:.1f}s")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio
            audio_file = os.path.join(temp_dir, "audio.wav")
            print("Extracting audio...")
            video.audio.write_audiofile(audio_file, logger=None)
            
            # Transcribe with Whisper
            print("Transcribing with Whisper...")
            result = whisper_model.transcribe(
                audio_file, 
                language='en',
                word_timestamps=True
            )
            
            segments = result.get("segments", [])
            if not segments:
                raise ValueError("No speech found in video")
            
            print(f"Found {len(segments)} speech segments")
            
            # Process and validate segments
            valid_segments = []
            for segment in segments:
                start = max(0, segment.get("start", 0))
                end = min(video.duration, segment.get("end", 0))
                text = segment.get("text", "").strip()
                
                if text and end > start and (end - start) >= 0.5:
                    valid_segments.append({
                        "start": start,
                        "end": end,
                        "text": text,
                        "gender": detect_voice_gender(audio_file, start, min(end, start + 5))
                    })
            
            print(f"Processing {len(valid_segments)} valid segments")
            
            # Split long segments
            valid_segments = split_long_segments(valid_segments)
            print(f"After splitting: {len(valid_segments)} segments")
            
            if not valid_segments:
                raise ValueError("No valid segments to process")
            
            # Convert to Hinglish
            original_texts = [seg["text"] for seg in valid_segments]
            print("Converting to Hinglish...")
            hinglish_texts = convert_to_hinglish(original_texts)
            
            # Ensure same length
            if len(hinglish_texts) != len(valid_segments):
                hinglish_texts = original_texts
            
            # Generate TTS
            print("Generating speech...")
            tts_files = []
            voices = []
            
            for i, (segment, hinglish_text) in enumerate(zip(valid_segments, hinglish_texts)):
                tts_file = os.path.join(temp_dir, f"speech_{i:03d}.mp3")
                voice = get_voice_name(segment["gender"])
                tts_files.append(tts_file)
                voices.append(voice)
            
            # Run batch TTS
            print("Running TTS batch processing...")
            tts_results = run_tts_batch(hinglish_texts, voices, tts_files)
            
            # Create audio timeline
            print("Building audio timeline...")
            audio_clips = []
            
            for i, (segment, tts_file) in enumerate(zip(valid_segments, tts_files)):
                start_time = segment["start"]
                duration = segment["end"] - segment["start"]
                
                try:
                    # Load TTS audio
                    if os.path.exists(tts_file):
                        tts_audio, sr = librosa.load(tts_file, sr=22050)
                    else:
                        # Fallback to WAV if MP3 failed
                        wav_file = tts_file.replace('.mp3', '.wav')
                        if os.path.exists(wav_file):
                            tts_audio, sr = librosa.load(wav_file, sr=22050)
                        else:
                            # Create silence
                            tts_audio = np.zeros(int(22050 * duration))
                            sr = 22050
                    
                    # Stretch audio to match segment duration
                    stretched_audio = stretch_audio_to_duration(tts_audio, sr, duration)
                    
                    # Save processed audio
                    wav_file = os.path.join(temp_dir, f"segment_{i:03d}.wav")
                    sf.write(wav_file, stretched_audio, sr)
                    
                    # Create audio clip
                    audio_clip = AudioFileClip(wav_file)
                    audio_clip = audio_clip.with_start(start_time).with_duration(duration)
                    audio_clips.append(audio_clip)
                    
                except Exception as e:
                    print(f"Error processing segment {i}: {e}")
                    continue
            
            if not audio_clips:
                raise ValueError("No audio clips generated")
            
            # Combine audio
            print("Combining audio tracks...")
            final_audio = CompositeAudioClip(audio_clips)
            
            # Create final video
            print("Creating final video...")
            final_video = video.with_audio(final_audio)
            
            # Write output with optimized settings
            final_video.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                logger=None
            )
            
            # Cleanup
            final_video.close()
            final_audio.close()
            for clip in audio_clips:
                clip.close()
            
    except Exception as e:
        print(f"Video processing error: {e}")
        raise
        
    finally:
        if video:
            video.close()
    
    print("Video processing complete!")

# ── WEB INTERFACE ────────────────────────────────────────────

def get_home_html():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>🎤 Simple Hinglish Dubber</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f0f0; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        form { text-align: center; }
        input[type="file"] { margin: 20px 0; padding: 10px; }
        button { background: #007cba; color: white; padding: 15px 30px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        button:hover { background: #005a87; }
        .progress { display: none; margin: 20px 0; }
        .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #007cba; width: 0%; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Simple Hinglish Video Dubber</h1>
        <p>Upload a video and convert speech to Hinglish dubbing!</p>
        <form method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
            <input type="file" name="video" accept="video/*" required id="videoFile">
            <br><button type="submit" id="submitBtn">Convert to Hinglish</button>
        </form>
        <div class="progress" id="progressDiv">
            <p>Processing your video... This may take a few minutes.</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('uploadForm').onsubmit = function() {
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('submitBtn').textContent = 'Processing...';
            document.getElementById('progressDiv').style.display = 'block';
            
            // Fake progress bar
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress > 90) progress = 90;
                document.getElementById('progressFill').style.width = progress + '%';
            }, 2000);
        };
    </script>
</body>
</html>'''

def get_result_html(url):
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>✅ Done!</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f0f0f0; }}
        .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }}
        video {{ max-width: 100%; margin: 20px 0; }}
        a {{ display: inline-block; margin: 10px; padding: 10px 20px; background: #007cba; color: white; text-decoration: none; border-radius: 5px; }}
        a:hover {{ background: #005a87; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✅ Conversion Complete!</h1>
        <video src="{url}" controls></video><br>
        <a href="{url}" download>⬇️ Download Video</a>
        <a href="/">🔄 Convert Another</a>
    </div>
</body>
</html>'''

# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def home():
    return get_home_html()

@app.route("/upload", methods=["POST"])
def upload():
    print("Upload received!")
    
    try:
        if "video" not in request.files:
            return "No video file uploaded!", 400
        
        video_file = request.files["video"]
        if not video_file.filename:
            return "No file selected!", 400
        
        # Validate file type
        allowed_types = ["mp4", "mov", "mkv", "webm", "m4v", "avi"]
        filename = video_file.filename.lower()
        file_ext = filename.rsplit(".", 1)[1] if "." in filename else ""
        
        if file_ext not in allowed_types:
            return f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_types)}", 400
        
        # Generate unique filename
        file_id = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}.{file_ext}")
        output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")
        
        # Save uploaded file
        video_file.save(input_path)
        print(f"File saved: {input_path}")
        
        # Process video
        process_video_simple(input_path, output_path)
        
        # Return success page
        return get_result_html(f"/download/{file_id}.mp4")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Processing error: {error_msg}")
        traceback.print_exc()
        return f"Processing failed: {error_msg}", 500
    
    finally:
        # Cleanup input file
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass

@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(OUTPUT_DIR, filename)
    except FileNotFoundError:
        abort(404)

@app.errorhandler(413)
def file_too_large(e):
    return "File too large! Please upload a video smaller than 1.5GB.", 413

@app.errorhandler(404)
def not_found(e):
    return "File not found!", 404

@app.errorhandler(500)
def internal_error(e):
    return "Internal server error. Please try again.", 500

# ── STARTUP ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("-" * 50)
    print("🎤 Simple Hinglish Video Dubber - Fully Compatible")
    print("-" * 50)
    print("Starting server...")
    
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        threaded=True
    )
