import subprocess
import whisper
from transformers import pipeline
import os

def extract_audio_ffmpeg(video_path, audio_path='audio.wav'):
    if os.path.exists(audio_path):
        os.remove(audio_path)  # clean old audio
    command = [
        "ffmpeg", "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",  # sample rate
        "-ac", "1",  # mono
        audio_path
    ]
    subprocess.run(command, check=True)
    return audio_path

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text, max_chunk=1000):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")  # Smaller, faster version
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ''
    for chunk in chunks:
        summary_piece = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summary += summary_piece[0]['summary_text'] + ' '
    return summary.strip()

def summarize_video(video_path):
    print("Extracting audio...")
    audio_path = extract_audio_ffmpeg(video_path)
    
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    
    print("Summarizing transcript...")
    summary = summarize_text(transcript)
    
    return summary

if __name__ == "__main__":
    video_file = "sample_video.mp4"  # replace with your video filename
    summary = summarize_video(video_file)
    print("\n=== Video Summary ===\n", summary)
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
