"""
ingest.py — Transcript Fetcher + ChromaDB Ingestion Pipeline
=============================================================
Iteration-aware: reads CHUNK_SIZE and CHUNK_OVERLAP from .env
so we can swap between v1/v2/v3 without changing code.

Transcript strategy (in order of preference):
  1. youtube-transcript-api  (fastest, no download needed)
  2. yt-dlp --write-subs     (subtitles only, no audio)
  3. Raise clear error       (avoids hammering YT)
"""

import os
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()
import chromadb.telemetry.product.posthog as telemetry
telemetry.Posthog.capture = lambda *args, **kwargs: None
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 100))
VIDEOS_FILE    = "videos.json"
DELAY_BETWEEN  = 3   # seconds between YT requests to prevent hammering YT!

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any youtube.com or youtu.be URL."""
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from: {url}")


def clean_transcript(raw: str) -> str:
    """Remove filler words, excessive whitespace from transcript text."""
    # Remove music/sound annotations like [Music] [Applause]
    text = re.sub(r'\[.*?\]', '', raw)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Strategy 1: youtube-transcript-api ────────────────────────────────────────

def fetch_via_transcript_api(video_id: str) -> Optional[str]:
    """
    Try to get English transcript using youtube-transcript-api.
    Falls back to auto-generated English if manual not available.
    Returns None if not possible.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # New API in v1.0+ — fetch directly
        ytt_api = YouTubeTranscriptApi()
        
        try:
            # Try English first
            fetched = ytt_api.fetch(video_id, languages=['en'])
        except Exception:
            try:
                # Try any language
                fetched = ytt_api.fetch(video_id)
            except Exception:
                return None

        full_text = " ".join([entry.text for entry in fetched])
        log.info("  ✅ Found transcript")
        return clean_transcript(full_text)

    except Exception as e:
        log.warning(f"  ⚠️  transcript-api failed: {e}")
        return None


# ── Strategy 2: yt-dlp (subtitles only, no audio download) ───────────────────

def fetch_via_ytdlp(video_id: str, output_dir: str = "./data/subs") -> Optional[str]:
    """
    Use yt-dlp to download subtitles ONLY (no video/audio) WITHOUT CREDENTIALS.
    """
    try:
        import yt_dlp
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            'skip_download': True,           # no video download
            'writesubtitles': True,          # write manual subs
            'writeautomaticsub': True,       # write auto subs if no manual
            'subtitleslangs': ['en'],        # English only
            'subtitlesformat': 'vtt',        # WebVTT format
            'outtmpl': f'{output_dir}/{video_id}',
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        # Find the downloaded .vtt file
        vtt_files = list(Path(output_dir).glob(f"{video_id}*.vtt"))
        if not vtt_files:
            log.warning("  ⚠️  yt-dlp: no .vtt subtitle file found")
            return None

        # Parse VTT file into plain text
        raw = vtt_files[0].read_text(encoding='utf-8')
        # Remove VTT formatting headers and timestamps
        text = re.sub(r'WEBVTT.*?\n\n', '', raw, flags=re.DOTALL)
        text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> .*\n', '', text)
        text = re.sub(r'<[^>]+>', '', text)   # remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()

        log.info("  ✅ yt-dlp subtitle fetch successful")
        return clean_transcript(text)

    except Exception as e:
        log.warning(f"  ⚠️  yt-dlp failed: {e}")
        return None


# ── Main fetch function with fallback chain ───────────────────────────────────

def fetch_transcript(url: str) -> Optional[str]:
    """
    Fetch transcript using strategy chain:
      1. youtube-transcript-api
      2. yt-dlp subtitles only
      3. Return None (log warning)
    """
    video_id = extract_video_id(url)
    log.info(f"  Trying youtube-transcript-api for {video_id}...")
    transcript = fetch_via_transcript_api(video_id)

    if transcript:
        return transcript

    log.info(f"  Falling back to yt-dlp for {video_id}...")
    transcript = fetch_via_ytdlp(video_id)

    if transcript:
        return transcript

    log.error(f"  ❌ Could not fetch transcript for {url}")
    return None


# ── ChromaDB ingestion ────────────────────────────────────────────────────────

def build_documents(video_meta: dict, transcript: str) -> list[Document]:
    """
    Split transcript into chunks and wrap each in a LangChain Document
    with rich metadata for retrieval + slide 4 of presentation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    chunks = splitter.split_text(transcript)
    log.info(f"  📄 Split into {len(chunks)} chunks "
             f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    video_id = extract_video_id(video_meta["url"])
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                # Core metadata
                "video_id":    video_id,
                "video_url":   video_meta["url"],
                "video_title": video_meta["title"],
                "channel":     video_meta["channel"],
                "cuisine":     video_meta["cuisine"],
                "tags":        ", ".join(video_meta.get("tags", [])),
                # Chunk position (useful for ordering retrieved context)
                "chunk_index": i,
                "total_chunks": len(chunks),
                # Iteration tracking
                "chunk_size":    CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        )
        documents.append(doc)
    return documents


def ingest_videos(videos: list[dict], reset_db: bool = False) -> Chroma:
    """
    Main ingestion pipeline:
    1. Load video list
    2. Fetch transcripts (with fallback)
    3. Chunk + embed
    4. Store in ChromaDB

    Args:
        videos:   list of video metadata dicts from videos.json
        reset_db: if True, wipe the DB first (useful between iterations)
    """
    log.info(f"🚀 Starting ingestion | chunk_size={CHUNK_SIZE} | overlap={CHUNK_OVERLAP}")
    log.info(f"📦 ChromaDB path: {CHROMA_DB_PATH}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Optionally wipe DB between iterations
    if reset_db:
        import shutil
        if Path(CHROMA_DB_PATH).exists():
            shutil.rmtree(CHROMA_DB_PATH)
            log.info("🗑️  Wiped existing ChromaDB")

    # Load or create ChromaDB
    vectorstore = Chroma(
        collection_name="asian_cooking",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    # Check which videos already ingested (avoid re-embedding)
    existing = vectorstore.get()
    existing_ids = set()
    if existing and existing.get("metadatas"):
        existing_ids = {m.get("video_id") for m in existing["metadatas"] if m}

    log.info(f"📊 Already in DB: {len(existing_ids)} video(s)")

    success, skipped, failed = 0, 0, 0

    for video in tqdm(videos, desc="Ingesting videos"):
        video_id = extract_video_id(video["url"])

        if video_id in existing_ids:
            log.info(f"⏭️  Skipping (already in DB): {video['title']}")
            skipped += 1
            continue

        log.info(f"\n🎬 Processing: {video['title']} [{video['channel']}]")

        transcript = fetch_transcript(video["url"])

        if not transcript:
            log.error(f"  ❌ Skipping — no transcript available")
            failed += 1
            # Respect YT — wait even on failure
            time.sleep(DELAY_BETWEEN)
            continue

        log.info(f"  📝 Transcript length: {len(transcript)} chars")

        documents = build_documents(video, transcript)
        vectorstore.add_documents(documents)

        success += 1
        log.info(f"  ✅ Ingested {len(documents)} chunks")

        # Be polite to YouTube — don't hammer!
        time.sleep(DELAY_BETWEEN)

    log.info(f"\n{'='*50}")
    log.info(f"✅ Done! Success: {success} | Skipped: {skipped} | Failed: {failed}")
    log.info(f"📦 Total docs in DB: {vectorstore._collection.count()}")

    return vectorstore


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest YouTube cooking videos into ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe ChromaDB before ingesting (use between iterations)")
    parser.add_argument("--videos", default=VIDEOS_FILE,
                        help="Path to videos.json file")
    parser.add_argument("--single", type=str, default=None,
                        help="Ingest a single video URL instead of the full list")
    args = parser.parse_args()

    if args.single:
        # Quick test with one URL
        videos = [{
            "url": args.single,
            "title": "Custom Video",
            "channel": "Custom",
            "cuisine": "Asian",
            "tags": []
        }]
    else:
        with open(args.videos, "r") as f:
            videos = json.load(f)

    ingest_videos(videos, reset_db=args.reset)
