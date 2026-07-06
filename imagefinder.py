#!/usr/bin/env python3
# =============================================================================
#  ImageFinder
#  -------------------------------------------------------------------------
#  A single-file local web app that lets you search a folder of IMAGES by
#  typing a phrase ("girl wearing jeans", "showing off tummy", "glasses") and
#  get back a grid of matching pictures. You select the ones you want and copy
#  the ORIGINAL source files into a destination folder.
#
#  This is the image sibling of ClipFinder. Images are much simpler than video
#  because there are no frames to extract and no clips to preview - each image
#  is its own thing. So the whole machine is just:
#
#  HOW IT WORKS (the whole machine in 5 lines):
#    1. THUMBNAIL: for each image, we make ONE small copy for the grid to show
#       (so we're not loading giant 40MB originals into a page of hundreds).
#    2. EMBED: CLIP turns each image into a vector (a list of numbers that
#       captures "what's in the picture"). We store these vectors.
#    3. SEARCH: your phrase is turned into a vector by the same CLIP model,
#       then compared to every image vector. Closest = best match.
#    4. BROWSE: with no search typed, we just show every image, so you can
#       scroll and copy without searching at all.
#    5. COPY: you tick the images you want, click copy, the ORIGINAL files are
#       duplicated into your destination folder.
#
#  This file IS the web server (FastAPI) AND serves the web page (HTML/JS).
#  Run it, it opens your browser, you do everything from there.
#
#  RUNS ON BOTH WINDOWS AND MAC:
#    - On Windows with an NVIDIA card it uses the GPU (CUDA).
#    - On Apple Silicon Macs (M1/M2/M3/M4) it uses the GPU (MPS).
#    - Anything else quietly falls back to CPU (still works, just slower).
# =============================================================================

# ----- standard library imports (these ship with Python, nothing to install) -
import os                # file paths, listing folders
import sys               # to detect platform / exit
import json              # saving + loading the index
import time              # timestamps
import shutil            # copying files
import hashlib           # making a quick fingerprint of a file (to detect changes)
import threading         # to open the browser after the server starts
import webbrowser        # to pop open your browser automatically
import subprocess        # to open files/folders in the OS
from pathlib import Path # nicer path handling than raw strings

# ----- third-party imports (installed via the pip line in the setup note) ----
# If any of these fail, the script prints a friendly message telling you what
# to install, instead of a confusing traceback.
try:
    import torch                          # the deep-learning engine (uses your GPU)
    import open_clip                      # the CLIP model (image+text understanding)
    from PIL import Image, ImageOps       # loading + thumbnailing images
    import numpy as np                    # fast math on the vectors
    from fastapi import FastAPI, Request  # the web backend framework
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    import uvicorn                        # the actual web server that runs FastAPI
except ImportError as e:
    print("\n[ImageFinder] A required package is missing:", e)
    print("Run ONE of these once in your terminal, then start the app again:\n")
    print("  # Windows with an NVIDIA GPU:")
    print('  pip install "fastapi" "uvicorn[standard]" "open_clip_torch" '
          '"torch --index-url https://download.pytorch.org/whl/cu121" '
          '"pillow" "pillow-heif" "numpy"\n')
    print("  # Mac (Apple Silicon or Intel) - plain torch, no CUDA line:")
    print('  pip install "fastapi" "uvicorn[standard]" "open_clip_torch" '
          '"torch" "pillow" "pillow-heif" "numpy"\n')
    print("(pillow-heif lets it open iPhone .heic photos. It's optional but handy.)")
    sys.exit(1)

# HEIC/HEIF support (iPhone photos). Optional: if the plugin isn't installed we
# just skip those files instead of crashing.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False


# =============================================================================
#  CONFIGURATION / CONSTANTS
#  These are the knobs. Tweak here if you want different defaults.
# =============================================================================

APP_PORT = 8001                      # the app will live at http://localhost:8001
INDEX_FILENAME = "imagefinder_index.json"  # where we save what we've indexed
THUMB_MAX = 400                      # thumbnail longest side in pixels (grid tiles)
DEFAULT_PER_PAGE = 24                # how many image tiles show per page by default

# How many images to embed at once during indexing. Bigger = faster, but uses
# more memory. 16 is a safe default that speeds up CPU a lot and works on a 6GB
# GPU. If you have a strong GPU with lots of VRAM, try 32 or 64. If you ever get
# an "out of memory" error while indexing, lower this (e.g. 8 or 4).
INDEX_BATCH_SIZE = 16

# Which file types we treat as images. (.heic needs the pillow-heif plugin.)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif",
                    ".bmp", ".tiff", ".tif", ".heic", ".heif"}

# Where we remember your folder paths between launches. Saved next to this
# script so the boxes auto-fill next time you open the app.
SETTINGS_FILENAME = Path(__file__).parent / "imagefinder_settings.json"

# CLIP model choice.
#   ViT-L-14  = higher accuracy, the sweet spot for a 6GB card like a 1660 Super.
#               Slower to INDEX (bigger model), but search itself stays fast and
#               results are noticeably more precise. This is the default.
#   ViT-B-32  = the lighter/faster fallback. If indexing ever feels too slow on
#               your hardware, comment out the two L-14 lines below and uncomment
#               the two B-32 lines - that's the whole switch.
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"   # trained weights for ViT-L-14
# CLIP_MODEL_NAME = "ViT-B-32"
# CLIP_PRETRAINED = "laion2b_s34b_b79k"   # trained weights for ViT-B-32

# The size of the vectors the chosen model produces. ViT-L-14 = 768, ViT-B-32 = 512.
# Used only to make correctly-shaped empty arrays when the index is empty.
MODEL_VECTOR_DIM = 768 if CLIP_MODEL_NAME == "ViT-L-14" else 512


# =============================================================================
#  GLOBAL STATE
#  Things the whole app shares: the loaded model, the in-memory index, and a
#  small dict tracking indexing progress so the UI can show a progress bar.
# =============================================================================

STATE = {
    "model": None,            # the CLIP model (loaded once, reused)
    "preprocess": None,       # the function that prepares an image for CLIP
    "tokenizer": None,        # turns your search text into tokens for CLIP
    "device": None,           # "cuda" (NVIDIA), "mps" (Apple), or "cpu"

    # The index. Unlike ClipFinder (one record per FRAME), here it's simply
    # one record per IMAGE - much simpler, no collapsing needed.
    #   images  - maps a source image path -> its info (thumbnail, fingerprint,
    #             width/height, source folder, vector row number).
    #   image_vectors - numpy array, one row per image, in the same order the
    #             image paths appear in "order".
    #   order   - the list of image paths, one per row of image_vectors, so we
    #             know which vector belongs to which image.
    "images": {},             # {image_path: {...}}
    "image_vectors": None,    # numpy array, shape (num_images, vector_size)
    "order": [],              # [image_path, ...] aligned with image_vectors rows

    # Folders the user picked in the UI (filled in from the frontend).
    #  sources  - a LIST of source folders. Indexing walks all of them (and
    #             their subfolders) into one shared library; the UI can then
    #             search/browse either ALL of them or just one at a time.
    #  thumbs/destination - single shared folders.
    "folders": {"sources": [], "thumbs": "", "destination": ""},

    # Images copied to the destination THIS SESSION. Used to grey out already-
    # copied tiles so you don't duplicate. Deliberately NOT saved to disk: it
    # resets when the server restarts, so cross-session duplicates are allowed.
    "copied_this_session": set(),

    # Live progress for the "Build/update index" button.
    "progress": {"running": False, "done": 0, "total": 0,
                 "message": "Idle", "cancel": False, "finished": False},
}


# =============================================================================
#  MODEL LOADING
#  Load CLIP once, on the best available device. Called lazily the first time
#  we need it (so the app window opens instantly and only loads the model when
#  you actually index or search).
# =============================================================================

def pick_device():
    """Choose the fastest thing available, in order: NVIDIA GPU (cuda),
    Apple Silicon GPU (mps), then plain CPU. This is what makes the SAME file
    use the GPU on both a Windows NVIDIA machine and an Apple Silicon Mac."""
    if torch.cuda.is_available():
        return "cuda"
    # getattr guards against very old torch builds that lack the mps attribute.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_if_needed():
    if STATE["model"] is not None:
        return  # already loaded, nothing to do

    STATE["device"] = pick_device()
    dev = STATE["device"]
    print("=" * 70)
    if dev == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "NVIDIA GPU"
        print(f"[ImageFinder] Running on GPU (CUDA): {name}")
    elif dev == "mps":
        print("[ImageFinder] Running on GPU (Apple Silicon / MPS).")
    else:
        print("[ImageFinder] Running on CPU (no supported GPU detected).")
        print("  The app works fine, but indexing will be slower.")
    print("=" * 70)

    print(f"[ImageFinder] Loading model '{CLIP_MODEL_NAME}' (first run downloads it)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model = model.to(dev).eval()   # move to device, set to inference mode
    STATE["model"] = model
    STATE["preprocess"] = preprocess
    STATE["tokenizer"] = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    print("[ImageFinder] Model ready.")


# =============================================================================
#  HELPERS: fingerprints, thumbnails, embedding
# =============================================================================

def file_fingerprint(path: Path) -> str:
    """A cheap, reliable 'has this file changed?' signature: size + mtime.
    Lets us skip re-indexing images that haven't changed since last time."""
    stat = path.stat()
    raw = f"{stat.st_size}-{int(stat.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()


def open_image(path: Path) -> Image.Image:
    """Open an image and fix its rotation. Phone photos store orientation in
    EXIF instead of actually rotating the pixels; exif_transpose applies it so
    thumbnails aren't sideways. Always returns RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # honor phone rotation
    return img.convert("RGB")


def make_thumbnail_from_image(img, image: Path, thumbs_dir: Path):
    """Make ONE small copy for the grid from an ALREADY-OPENED image, and report
    the original's pixel size. Returns (thumb_path, width, height), or
    (None, 0, 0) if saving fails.

    This is the fast path used during indexing: we open each file only once and
    reuse that same open image for both the thumbnail and the embedding, instead
    of decoding the file twice.

    Why a thumbnail at all? A page showing hundreds of full-size originals would
    be enormous and slow. The thumbnail is the only image the grid ever loads."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.md5(str(image).encode()).hexdigest()[:16]
    out_path = thumbs_dir / f"{safe_name}.jpg"
    try:
        w, h = img.size                       # the ORIGINAL resolution (for the tag)
        # thumbnail() shrinks in place, keeping aspect ratio, never upscaling.
        thumb = img.copy()
        thumb.thumbnail((THUMB_MAX, THUMB_MAX))
        thumb.save(out_path, "JPEG", quality=85)
        return out_path, w, h
    except Exception:
        return None, 0, 0


def make_thumbnail(image: Path, thumbs_dir: Path):
    """Open an image from disk and make its thumbnail. Kept for any caller that
    only has a path; the indexing loop uses make_thumbnail_from_image instead so
    it doesn't open the same file twice."""
    try:
        img = open_image(image)
    except Exception:
        return None, 0, 0
    return make_thumbnail_from_image(img, image, thumbs_dir)


def embed_images(image_paths: list[Path]) -> np.ndarray:
    """Run a batch of images through CLIP to get their vectors. Returns a numpy
    array, one row per image. Done in batches so the GPU stays busy without
    running out of memory. Images that fail to open are skipped, so the caller
    must pair results back up carefully - see embed_one_image below for the
    per-image version we actually use during indexing."""
    load_model_if_needed()
    vectors = []
    batch = []
    batch_size = 32

    def flush(batch_imgs):
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(STATE["device"])
        with torch.no_grad():   # we're not training, so skip gradient tracking
            feats = STATE["model"].encode_image(tensor)
            # Normalize so comparing vectors = comparing directions (cosine).
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.cpu().numpy())

    for p in image_paths:
        try:
            img = open_image(p)
        except Exception:
            continue
        batch.append(STATE["preprocess"](img))
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    if not vectors:
        return np.zeros((0, MODEL_VECTOR_DIM), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32)


def embed_one_image(image: Path):
    """Embed a SINGLE image and return its vector (or None if it can't be read).
    We use this during indexing so each image is handled independently - if one
    image is corrupt we just skip that one, and every other image still lines up
    correctly with its own vector."""
    load_model_if_needed()
    try:
        img = open_image(image)
    except Exception:
        return None
    tensor = STATE["preprocess"](img).unsqueeze(0).to(STATE["device"])
    with torch.no_grad():
        feats = STATE["model"].encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]   # single vector


def embed_pil_batch(images: list) -> np.ndarray:
    """Embed a batch of ALREADY-OPENED images (PIL Image objects) in ONE pass
    through the model. This is the fast path used during indexing: instead of
    running the model once per image, we hand it a whole stack at once, which is
    dramatically more efficient on both GPU and CPU.

    Returns a numpy array with one row per input image, in the SAME order the
    images were given. The caller is responsible for only passing images it has
    already successfully opened, so rows line up with paths one-to-one."""
    load_model_if_needed()
    if not images:
        return np.zeros((0, MODEL_VECTOR_DIM), dtype=np.float32)
    # Preprocess each image (resize/normalize the way CLIP expects), then stack
    # them into a single tensor and send the whole batch to the model together.
    tensors = [STATE["preprocess"](img) for img in images]
    tensor = torch.stack(tensors).to(STATE["device"])
    with torch.no_grad():   # not training, so skip gradient tracking (faster)
        feats = STATE["model"].encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # normalize for cosine
    return feats.cpu().numpy().astype(np.float32)


def embed_text(query: str) -> np.ndarray:
    """Turn the search phrase into a CLIP vector, the same 'shape' as the image
    vectors, so we can compare them directly."""
    load_model_if_needed()
    tokens = STATE["tokenizer"]([query]).to(STATE["device"])
    with torch.no_grad():
        feats = STATE["model"].encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]   # single vector


# =============================================================================
#  PATH HELPERS
# =============================================================================

def canon_path(p: str) -> str:
    """Return a canonical form of a folder path so the SAME folder always
    compares equal, no matter how it was typed. Windows is case-insensitive and
    accepts both slash styles, so 'C:/pics/X', 'C:\\pics\\X\\' and 'C:\\Pics\\X'
    all collapse to one string. Empty stays empty."""
    if not p:
        return ""
    try:
        return os.path.normcase(os.path.normpath(str(p)))
    except Exception:
        return str(p)


# =============================================================================
#  INDEX SAVE / LOAD
#  We persist the index to a JSON file in the thumbnails folder so you don't
#  have to re-index every launch. Vectors are saved alongside as a .npy
#  (numpy's fast binary format) because JSON is bad at big number arrays.
# =============================================================================

def index_paths():
    base = Path(STATE["folders"]["thumbs"] or ".")
    return base / INDEX_FILENAME, base / "imagefinder_vectors.npy"


def save_settings():
    """Write the current folder paths to the settings file."""
    try:
        SETTINGS_FILENAME.write_text(json.dumps(STATE["folders"]))
    except Exception as e:
        print(f"[ImageFinder] Could not save settings: {e}")


def load_settings():
    """Read the saved folder paths from the settings file, if it exists, and
    pre-fill them. Called once at startup. Also migrates an older single
    'source' key to the new 'sources' list."""
    try:
        if SETTINGS_FILENAME.exists():
            saved = json.loads(SETTINGS_FILENAME.read_text())
            if isinstance(saved.get("sources"), list):
                STATE["folders"]["sources"] = [s for s in saved["sources"] if s]
            elif saved.get("source"):
                STATE["folders"]["sources"] = [saved["source"]]
            for k in ("thumbs", "destination"):
                if saved.get(k):
                    STATE["folders"][k] = saved[k]
            print("[ImageFinder] Loaded saved folder paths.")
    except Exception as e:
        print(f"[ImageFinder] Could not load settings: {e}")


def save_index():
    meta_path, vec_path = index_paths()
    meta = {
        "images": STATE["images"],
        "order": STATE["order"],
        "folders": STATE["folders"],
    }
    meta_path.write_text(json.dumps(meta))
    if STATE["image_vectors"] is not None:
        np.save(vec_path, STATE["image_vectors"])


def load_index():
    meta_path, vec_path = index_paths()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        STATE["images"] = meta.get("images", {})
        STATE["order"] = meta.get("order", [])
        saved_folders = meta.get("folders", {})
        if "sources" not in saved_folders and saved_folders.get("source"):
            saved_folders = dict(saved_folders)
            saved_folders["sources"] = [saved_folders["source"]]
        for k, v in saved_folders.items():
            if k == "source":
                continue
            if k == "sources":
                if not STATE["folders"].get("sources"):
                    STATE["folders"]["sources"] = [s for s in (v or []) if s]
            elif not STATE["folders"].get(k):
                STATE["folders"][k] = v
    if vec_path.exists():
        loaded = np.load(vec_path)
        # SAFETY: if this index was built with a DIFFERENT model, its vectors are
        # the wrong size (e.g. an old ViT-B-32 index = 512 dims vs ViT-L-14 = 768).
        # Mixing sizes would crash searches, so we discard the stale index and
        # tell the user to rebuild rather than load incompatible data.
        if loaded.ndim != 2 or loaded.shape[1] != MODEL_VECTOR_DIM:
            print(f"[ImageFinder] Existing index was built with a different model "
                  f"(vector size mismatch). Ignoring it - click "
                  f"'Build / update index' to rebuild.")
            STATE["image_vectors"] = None
            STATE["order"] = []
            STATE["images"] = {}
        else:
            STATE["image_vectors"] = loaded


# =============================================================================
#  THE INDEXING JOB (incremental)
#  This is what the "Build/update index" button triggers. It runs in a
#  background thread so the web page stays responsive and can poll progress.
# =============================================================================

def build_index_job():
    p = STATE["progress"]
    p.update({"running": True, "finished": False, "cancel": False,
              "done": 0, "total": 0, "message": "Scanning folders\u2026"})
    try:
        thumbs_dir = Path(STATE["folders"]["thumbs"])

        # Gather every source folder the user added, canonicalized + de-duped.
        raw_sources = [s for s in STATE["folders"].get("sources", []) if s]
        source_folders = []
        seen_src = set()
        for s in raw_sources:
            c = canon_path(s)
            if c and c not in seen_src:
                seen_src.add(c)
                source_folders.append(c)
        if not source_folders:
            p.update({"message": "No source folders set. Add at least one, "
                                 "click Save folders, then index."})
            return

        # Find every image across ALL source folders, INCLUDING SUBFOLDERS
        # (that's what rglob does - it walks the whole tree recursively).
        all_images = []           # list of (image_path, source_folder_str)
        for sf in source_folders:
            base = Path(sf)
            if not base.exists():
                continue
            for f in base.rglob("*"):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    # Skip .heic/.heif if we can't actually open them.
                    if f.suffix.lower() in (".heic", ".heif") and not HEIC_OK:
                        continue
                    all_images.append((f, sf))

        # If the same image shows up under two overlapping source folders,
        # keep the first source we saw it under.
        seen_paths = set()
        deduped = []
        for v, sf in all_images:
            if str(v) in seen_paths:
                continue
            seen_paths.add(str(v))
            deduped.append((v, sf))
        all_images = deduped

        # INCREMENTAL: figure out which images are new or changed since last time.
        to_process = []
        for v, sf in all_images:
            key = str(v)
            fp = file_fingerprint(v)
            known = STATE["images"].get(key)
            if (known is None) or (known.get("fingerprint") != fp):
                to_process.append((v, fp, sf))
            elif known.get("source") != sf:
                known["source"] = sf   # cheap correction, no reprocessing needed

        # PRUNE: drop images that no longer exist on disk.
        existing_keys = {str(v) for v, _sf in all_images}
        removed = [k for k in STATE["images"] if k not in existing_keys]
        for k in removed:
            STATE["images"].pop(k, None)

        total = len(to_process)
        skipped = len(all_images) - total
        p.update({"done": 0, "total": total,
                  "message": f"Starting: {total} new/changed to index "
                             f"({skipped} unchanged)"})

        # Process images in BATCHES. Instead of running the model once per image
        # (slow), we gather up to INDEX_BATCH_SIZE images and embed them all in
        # one pass, which is far faster on both GPU and CPU.
        #
        # For each image we open it ONCE and reuse that single open for both the
        # thumbnail AND the embedding, so we're not decoding every file twice.
        #
        # Cancel still works: we check the flag between batches, and everything
        # already finished is kept (indexing is incremental, so a later run
        # resumes from where we stopped).
        cancelled = False

        def flush_batch(pending):
            """Embed one batch of opened images and store their results.
            'pending' is a list of dicts, each holding the opened image plus the
            bookkeeping (path, fingerprint, source, thumbnail path, size)."""
            if not pending:
                return
            imgs = [item["img"] for item in pending]
            vecs = embed_pil_batch(imgs)          # ONE pass for the whole batch
            for item, vec in zip(pending, vecs):
                STATE["images"][str(item["image"])] = {
                    "thumb": item["thumb"],
                    "fingerprint": item["fp"],
                    "width": item["w"],
                    "height": item["h"],
                    "source": item["sf"],
                    "vector": vec.tolist(),   # stored temporarily; matrix rebuilt below
                }

        pending = []          # images opened + thumbnailed, waiting to be embedded
        for i, (image, fp, sf) in enumerate(to_process):
            # Check cancel BETWEEN images, but only act on it at a batch boundary
            # so we don't drop a half-prepared batch.
            if p.get("cancel"):
                cancelled = True
                break

            p["message"] = f"Indexing {i+1}/{total}: {image.name}"
            p["done"] = i   # count reflects images finished before this one

            # Open the image ONCE. If it can't be opened, skip it entirely.
            try:
                img = open_image(image)
            except Exception:
                continue

            # 1) thumbnail (made from the already-open image) + original size
            thumb, w, h = make_thumbnail_from_image(img, image, thumbs_dir)
            if thumb is None:
                continue

            # 2) queue this image for the next batch embedding
            pending.append({"img": img, "image": image, "fp": fp, "sf": sf,
                            "thumb": str(thumb), "w": w, "h": h})

            # When the batch is full, embed it all at once, then start a new one.
            if len(pending) >= INDEX_BATCH_SIZE:
                flush_batch(pending)
                pending = []
                p["done"] = i + 1

        # Embed whatever's left in the final partial batch (unless we cancelled
        # mid-way, in which case we still keep it - it's finished work).
        flush_batch(pending)
        if not cancelled:
            p["done"] = total

        # Rebuild the master vector matrix from ALL images we currently know
        # about (unchanged ones kept their old vector row; new ones just got a
        # fresh "vector" field above). This keeps rows and paths perfectly aligned.
        rebuild_vector_matrix()
        save_index()

        done_count = sum(1 for _ in to_process) if not cancelled else i
        total_in_library = len(STATE["images"])
        if cancelled:
            p["message"] = (f"Cancelled. {total_in_library} total in library.")
        else:
            p["message"] = (f"Done. Indexed {total} this run. "
                            f"{total_in_library} total in library.")
    except Exception as e:
        p["message"] = f"Error: {e}"
    finally:
        p["running"] = False
        p["finished"] = True
        p["cancel"] = False


def rebuild_vector_matrix():
    """Rebuild the big vector matrix (and the aligned 'order' list) from the
    per-image vectors we have stored. Newly-indexed images carry a temporary
    'vector' field; unchanged images keep the row they already had in the old
    matrix. This keeps every row lined up with exactly one image path.

    Kept deliberately straightforward over clever: we just walk every image,
    grab its vector (fresh field if present, else its old matrix row), and
    stack them all into one array."""
    old_matrix = STATE["image_vectors"]
    old_order = STATE["order"]
    # Map an image path -> its row index in the OLD matrix, so unchanged images
    # can reuse their existing vector without recomputing.
    old_row_of = {path: i for i, path in enumerate(old_order)}

    rows = []
    new_order = []
    for path, info in STATE["images"].items():
        if "vector" in info and info["vector"] is not None:
            # Freshly indexed this run - use the vector we just computed.
            rows.append(np.asarray(info["vector"], dtype=np.float32))
            del info["vector"]   # don't keep big vectors in the JSON metadata
        elif old_matrix is not None and path in old_row_of:
            # Unchanged - reuse the row it had before.
            rows.append(old_matrix[old_row_of[path]])
        else:
            # No vector anywhere (shouldn't happen) - skip to stay consistent.
            continue
        new_order.append(path)

    STATE["image_vectors"] = (np.vstack(rows).astype(np.float32)
                              if rows else np.zeros((0, MODEL_VECTOR_DIM),
                                                    dtype=np.float32))
    STATE["order"] = new_order


# =============================================================================
#  SEARCH + BROWSE
# =============================================================================

def video_in_scope(image: str, source: str = "") -> bool:
    """Decide whether an image belongs in the current view. source="" (or
    "all") means the whole pooled library. Otherwise we only keep images whose
    stored 'source' tag matches the chosen folder, with a path-prefix fallback
    for older indexes."""
    if not source or source == "all":
        return True
    source = canon_path(source)
    info = STATE["images"].get(image, {})
    tagged = info.get("source")
    if tagged is not None:
        return canon_path(tagged) == source
    try:
        return canon_path(source) in {canon_path(str(par))
                                      for par in Path(image).parents}
    except Exception:
        return False


def build_result_item(image: str, score=None):
    """Make one result dict for the grid from an image path. Shared by search
    (which passes a score) and browse-all (no score)."""
    info = STATE["images"].get(image)
    if not info:
        return None
    w, h = info.get("width", 0), info.get("height", 0)
    return {
        "image": image,                         # original source path (for copying/opening)
        "thumb": info["thumb"],                 # the small image the grid shows
        "name": Path(image).name,
        "score": round(float(score), 3) if score is not None else None,
        "width": w,
        "height": h,
        "resolution": f"{w}\u00d7{h}" if w and h else "",   # e.g. "4032×3024"
        "megapixels": round((w * h) / 1_000_000, 1) if w and h else 0,
        "copied": image in STATE["copied_this_session"],
    }


def search(query: str, max_images: int = 0, source: str = ""):
    """Search indexed images for the query and return them ranked best-first.
    max_images=0 means NO cap. source="" searches the whole library; a folder
    path limits it to that one source folder."""
    if STATE["image_vectors"] is None or len(STATE["image_vectors"]) == 0:
        return []

    qvec = embed_text(query)                     # (vector_size,)
    # Cosine similarity = dot product, because everything is normalized.
    scores = STATE["image_vectors"] @ qvec       # (num_images,)

    # Pair each image path with its score, filter by scope, sort best-first.
    scored = []
    for path, score in zip(STATE["order"], scores):
        if not video_in_scope(path, source):
            continue
        scored.append((path, float(score)))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    if max_images and max_images > 0:
        scored = scored[:max_images]

    results = []
    for image, score in scored:
        item = build_result_item(image, score)
        if item:
            results.append(item)
    return results


def browse_all(source: str = ""):
    """Return indexed images (no search), sorted by filename. The default view
    so you can browse and copy without typing a query."""
    images = sorted(STATE["images"].keys(), key=lambda v: Path(v).name.lower())
    results = []
    for image in images:
        if not video_in_scope(image, source):
            continue
        item = build_result_item(image, None)
        if item:
            results.append(item)
    return results


def indexed_sources():
    """The distinct source folders currently in the index, for the single-folder
    dropdown. Falls back to the saved sources list so folders show up even
    before anything's indexed. Deduped by canonical path."""
    found = []
    seen = set()

    def add(s):
        if not s:
            return
        key = canon_path(s)
        if key in seen:
            return
        seen.add(key)
        found.append(s)

    for info in STATE["images"].values():
        add(info.get("source"))
    for s in STATE["folders"].get("sources", []):
        add(s)
    return sorted(found, key=lambda p: Path(p).name.lower())


# =============================================================================
#  THE WEB APP (FastAPI)
#    GET  /                 -> the HTML page itself
#    GET  /get_folders      -> pre-fill the folder boxes
#    POST /set_folders      -> save the folder paths the user picked
#    GET  /browse_folder    -> native OS folder picker
#    POST /index            -> start the background indexing job
#    POST /cancel_index     -> ask a running job to stop
#    GET  /progress         -> the UI polls this for the progress bar
#    GET  /search?q=...     -> run a search, return ranked image results
#    GET  /browse_all       -> all indexed images (no search)
#    GET  /sources          -> source folders for the single-folder dropdown
#    GET  /thumb?path=..    -> stream a thumbnail image to the grid
#    POST /open_file        -> open the original in the default image viewer
#    POST /reveal_file      -> open the containing folder, file highlighted
#    POST /copy             -> copy selected originals to destination
# =============================================================================

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/set_folders")
async def set_folders(request: Request):
    data = await request.json()
    if "sources" in data and isinstance(data["sources"], list):
        STATE["folders"]["sources"] = [s.strip() for s in data["sources"] if s and s.strip()]
    for key in ("thumbs", "destination"):
        if key in data:
            STATE["folders"][key] = data[key].strip()
    save_settings()
    load_index()   # now that we know where thumbs live, try loading an index
    return {"ok": True, "folders": STATE["folders"]}


@app.get("/get_folders")
def get_folders():
    return {"folders": STATE["folders"]}


@app.get("/browse_folder")
def browse_folder():
    """Open a native folder-picker (via tkinter, which ships with Python) and
    return the chosen path. The web page can't open a real folder dialog itself
    due to browser security, so the backend does it. Works on Windows and Mac.

    Note: the dialog can sometimes open BEHIND the browser window - alt-tab
    (Windows) or Cmd-Tab (Mac) to it if you don't see it pop up."""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory()
        root.destroy()
        return {"ok": True, "path": folder or ""}
    except Exception as e:
        return {"ok": False, "path": "", "error": str(e)}


@app.post("/index")
def start_index():
    if STATE["progress"]["running"]:
        return {"ok": False, "message": "Already running."}
    threading.Thread(target=build_index_job, daemon=True).start()
    return {"ok": True}


@app.post("/cancel_index")
def cancel_index():
    if STATE["progress"]["running"]:
        STATE["progress"]["cancel"] = True
        STATE["progress"]["message"] = "Cancelling after current image\u2026"
        return {"ok": True}
    return {"ok": False, "message": "Nothing is running."}


@app.get("/progress")
def get_progress():
    return STATE["progress"]


@app.get("/search")
def do_search(q: str, source: str = ""):
    return JSONResponse(search(q, source=source))


@app.get("/browse_all")
def do_browse_all(source: str = ""):
    return JSONResponse(browse_all(source=source))


@app.get("/sources")
def do_sources():
    return {"sources": [{"path": s, "name": Path(s).name or s}
                        for s in indexed_sources()]}


@app.post("/open_file")
async def open_file(request: Request):
    """Open the ORIGINAL image in the computer's default image viewer."""
    data = await request.json()
    path = data.get("path", "")
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": "File not found."}
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))                       # Windows
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])         # macOS
        else:
            subprocess.Popen(["xdg-open", str(p)])     # Linux
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/reveal_file")
async def reveal_file(request: Request):
    """Open the folder CONTAINING the original image and highlight the file, so
    you can drag it straight into another app. Windows uses Explorer's /select;
    macOS uses Finder's -R reveal; Linux just opens the folder."""
    data = await request.json()
    path = data.get("path", "")
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": "File not found."}
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", "/select,", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent)])
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/thumb")
def get_thumb(path: str):
    # Serve a thumbnail by absolute path. We only serve .jpg files (the thumbs
    # we made), as a basic safety check.
    p = Path(path)
    if p.exists() and p.suffix.lower() in (".jpg", ".jpeg"):
        return FileResponse(str(p), media_type="image/jpeg")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/copy")
async def copy_selected(request: Request):
    data = await request.json()
    images = data.get("images", [])           # list of ORIGINAL source paths
    dest = Path(STATE["folders"]["destination"])
    dest.mkdir(parents=True, exist_ok=True)

    copied, skipped, copied_paths = [], [], []
    for v in images:
        src = Path(v)
        if not src.exists():
            skipped.append(v)
            continue
        target = dest / src.name
        # If a file with that name already exists, add a number so we don't clobber.
        if target.exists():
            stem, suffix = target.stem, target.suffix
            n = 1
            while (dest / f"{stem}_{n}{suffix}").exists():
                n += 1
            target = dest / f"{stem}_{n}{suffix}"
        shutil.copy2(src, target)             # copy2 preserves timestamps
        copied.append(str(target))
        STATE["copied_this_session"].add(v)
        copied_paths.append(v)
    return {"ok": True, "copied": len(copied), "skipped": len(skipped),
            "copied_images": copied_paths}


# =============================================================================
#  THE FRONTEND (HTML + CSS + JavaScript), served as one string.
# =============================================================================

HTML_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Image Finder</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background:#15171c; color:#e8e8ea; }
  header { padding:14px 18px; background:#1d2027; border-bottom:1px solid #2a2e37; }
  h1 { font-size:18px; margin:0; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding:12px 18px; }
  label { font-size:12px; color:#9aa0aa; display:block; margin-bottom:3px; }
  input[type=text]{ background:#0f1115; border:1px solid #2a2e37; color:#e8e8ea;
    padding:7px 9px; border-radius:6px; font-size:13px; }
  .folder input{ width:250px; }
  .pick{ display:flex; gap:4px; }
  button.browse{ background:#2a2e37; padding:7px 10px; font-size:12px; }
  .statusmsg{ font-size:13px; color:#4ade80; align-self:center; }
  button { background:#3b82f6; color:white; border:none; padding:8px 14px;
    border-radius:6px; font-size:13px; cursor:pointer; }
  button.secondary{ background:#2a2e37; }
  button.secondary.active{ background:#26303f; border:1px solid #3b82f6; }
  button:disabled{ opacity:.5; cursor:default; }
  #searchbar{ width:420px; font-size:15px; padding:10px 12px; }

  .setup-head{ display:flex; align-items:center; gap:8px; cursor:pointer;
    padding:8px 18px; color:#cfd3da; font-size:13px; user-select:none; }
  .setup-head:hover{ color:#fff; }
  .setup-head .chev{ transition:transform .15s; display:inline-block; }
  .setup-head.collapsed .chev{ transform:rotate(-90deg); }
  #setupBody.collapsed{ display:none; }
  #sourceList{ max-height:210px; overflow-y:auto; padding-right:4px;
    border:1px solid #2a2e37; border-radius:6px; padding:6px; background:#0f1115; }
  #sourceList:empty{ border:none; padding:0; }

  #progress{ font-size:12px; color:#9aa0aa; padding:0 18px 8px; }
  .bar{ height:6px; background:#2a2e37; border-radius:4px; overflow:hidden; margin-top:4px;}
  .bar > div{ height:100%; background:#3b82f6; width:0%; transition:width .3s; }

  #grid{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
    gap:10px; padding:14px 18px; }
  .tile{ position:relative; border:3px solid transparent; border-radius:8px;
    overflow:hidden; cursor:pointer; background:#0f1115; transition:box-shadow .12s, border-color .12s; }
  .tile.selected{ border-color:#3b82f6;
    box-shadow:0 0 0 2px #3b82f6, 0 0 14px 2px rgba(59,130,246,0.55); }
  .tile img{ width:100%; display:block; }
  .tile .meta{ font-size:11px; color:#9aa0aa; padding:6px 7px; display:flex;
    flex-direction:column; gap:5px; background:#0f1115; }
  .namerow{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .inforow{ display:flex; align-items:center; gap:5px; flex-wrap:wrap; }
  .check{ position:absolute; top:6px; right:6px; width:22px; height:22px;
    border-radius:50%; background:#3b82f6; color:white; display:none;
    align-items:center; justify-content:center; font-size:14px; }
  .tile.selected .check{ display:flex; }
  .tile.copied{ cursor:default; }
  .tile.copied img{ opacity:0.35; }
  .tile.copied .meta{ opacity:0.5; }
  .copiedtag{ position:absolute; top:6px; left:6px; background:#4ade80;
    color:#0f1115; font-size:10px; font-weight:700; padding:2px 6px;
    border-radius:4px; text-transform:uppercase; letter-spacing:.03em; }
  .pill{ font-size:10px; font-weight:600; padding:2px 7px; border-radius:999px;
    color:#fff; background:#2a2e37; white-space:nowrap; }
  .pill.likeness{ background:#3b82f6; }
  .pill.res{ background:#2a2e37; }
  .btngroup{ display:flex; gap:5px; margin-top:2px; }
  .openbtn{ flex:1; background:#2a2e37; color:#e8e8ea; border:none;
    padding:4px 9px; border-radius:5px; font-size:11px; cursor:pointer;
    white-space:nowrap; text-align:center; }
  .openbtn:hover{ background:#3b82f6; }

  .toolbar{ display:flex; gap:12px; align-items:center; padding:10px 18px;
    background:#1d2027; border-top:1px solid #2a2e37; position:sticky; bottom:0; }
  #count{ font-weight:600; }

  .srcrow{ display:flex; gap:4px; margin-bottom:6px; align-items:center; }
  .srcrow input{ width:340px; }
  .srcrow .removebtn{ background:#7f1d1d; padding:7px 10px; font-size:12px; }

  .tabs{ display:flex; gap:6px; align-items:center; padding:10px 18px 0; flex-wrap:wrap; }
  .tab{ background:#2a2e37; color:#cfd3da; border:1px solid #2a2e37; padding:7px 14px;
    border-radius:8px 8px 0 0; font-size:13px; cursor:pointer; }
  .tab.active{ background:#3b82f6; color:#fff; border-color:#3b82f6; }
  .tabs select{ background:#0f1115; border:1px solid #2a2e37; color:#e8e8ea;
    padding:7px 9px; border-radius:6px; font-size:13px; margin-left:6px; }
  .scopeinfo{ font-size:12px; color:#9aa0aa; margin-left:8px; }
</style>
</head>
<body>
<header><h1>Image Finder <span style="font-weight:400;color:#9aa0aa;font-size:14px">- by NovaPMV</span></h1></header>

<!-- ===== Folder setup (collapsible) ===== -->
<div id="setupHead" class="setup-head" onclick="toggleSetup()">
  <span class="chev">&#9660;</span>
  <span id="setupHeadLabel">Folder setup</span>
</div>
<div id="setupBody">

<div class="row">
  <div class="folder" style="flex:1 1 100%">
    <label>Source folders (each holds images — indexed together with all subfolders, searchable all-at-once or one at a time)</label>
    <div id="sourceList"></div>
    <button class="browse" style="margin-top:6px" onclick="addSourceRow('')">+ Add source folder</button>
  </div>
</div>
<div class="row">
  <div class="folder"><label>Thumbnails folder (small copies live here)</label>
    <div class="pick"><input id="f_thumbs" type="text" placeholder="C:\\imagefinder\\thumbs">
    <button class="browse" onclick="browse('f_thumbs')">Browse</button></div></div>
  <div class="folder"><label>Destination folder (copies go here)</label>
    <div class="pick"><input id="f_dest" type="text" placeholder="C:\\imagefinder\\selected">
    <button class="browse" onclick="browse('f_dest')">Browse</button></div></div>
  <div><label>&nbsp;</label><button onclick="saveFolders()">Save folders</button></div>
  <div><label>&nbsp;</label><button class="secondary" onclick="startIndex()">Build / update index</button></div>
  <div><label>&nbsp;</label><button id="cancelbtn" class="secondary" onclick="cancelIndex()" style="display:none;background:#7f1d1d">Cancel</button></div>
  <span id="folderstatus" class="statusmsg"></span>
</div>
</div><!-- /setupBody -->

<div id="progress">
  <span id="progtext">Idle</span>
  <div class="bar"><div id="barfill"></div></div>
</div>

<!-- ===== Scope tabs: search ALL folders, or a single chosen folder ===== -->
<div class="tabs">
  <div id="tabAll" class="tab active" onclick="setScopeAll()">All folders</div>
  <div id="tabOne" class="tab" onclick="setScopeOne()">Single folder</div>
  <select id="sourceSelect" style="display:none" onchange="onSourceSelectChange()"></select>
  <span id="scopeinfo" class="scopeinfo"></span>
</div>

<!-- ===== Controls row ===== -->
<div class="row" id="topnav" style="align-items:center">
  <div style="flex:1; display:flex; gap:10px; align-items:center; min-width:0;">
    <input id="searchbar" type="text" placeholder='Search e.g. "showing off tummy"'
           onkeydown="if(event.key==='Enter') runSearch()">
    <button onclick="runSearch()">Search</button>
    <button id="showCopiedBtn" class="secondary active" onclick="toggleShowCopied()">Disable Show Copied</button>
  </div>
  <div style="display:flex; gap:12px; align-items:center;">
    <button class="secondary" onclick="prevPage()">&larr; Prev</button>
    <span>page</span>
    <input id="pagenum" type="number" min="1" value="1" style="width:60px"
           onkeydown="if(event.key==='Enter') gotoPage()" onchange="gotoPage()">
    <span id="pageinfo">/ 0</span>
    <button class="secondary" onclick="nextPage()">Next &rarr;</button>
    <span id="totalinfo" class="scopeinfo"></span>
  </div>
  <div style="flex:1; display:flex; justify-content:flex-end; gap:12px; align-items:center;">
    <div><label>Per page</label>
      <input id="perpage" type="range" min="4" max="60" value="24"
             oninput="perpageLabel.textContent=this.value; renderPage()">
      <span id="perpageLabel">24</span></div>
    <div><label>Sort by</label>
      <select id="sortmode" onchange="applySort(); page=0; renderPage();">
        <option value="relevance">Relevance (search)</option>
        <option value="name">Filename (A\u2013Z)</option>
        <option value="name_desc">Filename (Z\u2013A)</option>
        <option value="res_desc">Resolution (high\u2192low)</option>
        <option value="res">Resolution (low\u2192high)</option>
      </select></div>
  </div>
</div>

<!-- ===== Results grid ===== -->
<div id="grid"></div>

<!-- ===== Bottom pagination ===== -->
<div class="row" id="bottomnav" style="justify-content:center">
  <button class="secondary" onclick="prevPage()">&larr; Prev</button>
  <span>page</span>
  <input id="pagenum_b" type="number" min="1" value="1" style="width:60px"
         onkeydown="if(event.key==='Enter') gotoPageBottom()" onchange="gotoPageBottom()">
  <span id="pageinfo_b">/ 0</span>
  <button class="secondary" onclick="nextPage()">Next &rarr;</button>
  <span id="totalinfo_b" class="scopeinfo"></span>
</div>

<!-- ===== Sticky bottom toolbar ===== -->
<div class="toolbar">
  <span id="count">0 selected</span>
  <button class="secondary" onclick="selectAllOnPage()">Select all on page</button>
  <button class="secondary" onclick="clearSelection()">Clear selection</button>
  <button onclick="copySelected()">Copy selected to destination</button>
  <span id="copystatus" class="statusmsg"></span>
</div>

<script>
let results = [];
let page = 0;
let selected = new Set();
let currentScope = "";
let lastQuery = "";

function renderSourceRows(paths){
  const list = document.getElementById('sourceList');
  list.innerHTML = '';
  if(!paths || !paths.length){ addSourceRow(''); return; }
  paths.forEach(p => addSourceRow(p));
}

function addSourceRow(value){
  const list = document.getElementById('sourceList');
  const row = document.createElement('div');
  row.className = 'srcrow';
  const input = document.createElement('input');
  input.type = 'text';
  input.placeholder = 'C:\\\\pics\\\\vacation';
  input.value = value || '';
  const browseBtn = document.createElement('button');
  browseBtn.className = 'browse';
  browseBtn.textContent = 'Browse';
  browseBtn.onclick = async () => {
    const r = await fetch('/browse_folder');
    const d = await r.json();
    if(d.ok && d.path){ input.value = d.path; }
    else if(!d.ok){ showStatus('folderstatus', 'Folder picker unavailable - type the path instead.', true); }
  };
  const removeBtn = document.createElement('button');
  removeBtn.className = 'browse removebtn';
  removeBtn.textContent = 'Remove';
  removeBtn.onclick = () => {
    row.remove();
    if(!document.querySelectorAll('#sourceList .srcrow').length) addSourceRow('');
  };
  row.appendChild(input);
  row.appendChild(browseBtn);
  row.appendChild(removeBtn);
  list.appendChild(row);
  if(!value){ list.scrollTop = list.scrollHeight; input.focus(); }
}

function toggleSetup(){
  const head = document.getElementById('setupHead');
  const body = document.getElementById('setupBody');
  const collapsed = body.classList.toggle('collapsed');
  head.classList.toggle('collapsed', collapsed);
  const label = document.getElementById('setupHeadLabel');
  if(collapsed){
    const n = getSourcePaths().length;
    label.textContent = 'Folder setup (' + n + (n === 1 ? ' source' : ' sources')
                        + ' — click to edit)';
  } else {
    label.textContent = 'Folder setup';
  }
}

function getSourcePaths(){
  return Array.from(document.querySelectorAll('#sourceList .srcrow input'))
    .map(i => i.value.trim())
    .filter(v => v.length);
}

window.addEventListener('load', async () => {
  try {
    const r = await fetch('/get_folders');
    const d = await r.json();
    const f = d.folders || {};
    renderSourceRows(f.sources || []);
    if(f.thumbs) f_thumbs.value = f.thumbs;
    if(f.destination) f_dest.value = f.destination;
  } catch(e) { renderSourceRows([]); }
  await refreshSourceDropdown();
  loadAllImages();
});

async function refreshSourceDropdown(){
  try {
    const r = await fetch('/sources');
    const d = await r.json();
    const sel = document.getElementById('sourceSelect');
    const prev = sel.value;
    sel.innerHTML = '';
    (d.sources || []).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.path; opt.textContent = s.name;
      sel.appendChild(opt);
    });
    if(prev && (d.sources||[]).some(s => s.path === prev)) sel.value = prev;
  } catch(e) { }
}

async function loadAllImages(){
  try {
    const r = await fetch('/browse_all?source=' + encodeURIComponent(currentScope));
    const all = await r.json();
    results = all || [];
    results.forEach((it,i) => it._ord = i);
    applySort();
    page = 0;
    copiedSet = new Set();
    renderPage();
  } catch(e) { }
}

function setScopeAll(){
  currentScope = "";
  document.getElementById('tabAll').classList.add('active');
  document.getElementById('tabOne').classList.remove('active');
  document.getElementById('sourceSelect').style.display = 'none';
  document.getElementById('scopeinfo').textContent = '';
  selected.clear();
  reapplyScope();
}
async function setScopeOne(){
  await refreshSourceDropdown();
  const sel = document.getElementById('sourceSelect');
  if(!sel.options.length){
    showStatus('folderstatus', 'No indexed folders yet - add sources and build the index first.', true);
    return;
  }
  document.getElementById('tabOne').classList.add('active');
  document.getElementById('tabAll').classList.remove('active');
  sel.style.display = '';
  currentScope = sel.value;
  document.getElementById('scopeinfo').textContent = 'Showing only this folder';
  selected.clear();
  reapplyScope();
}
function onSourceSelectChange(){
  currentScope = document.getElementById('sourceSelect').value;
  selected.clear();
  reapplyScope();
}
function reapplyScope(){
  if(lastQuery){ runSearch(true); }
  else { loadAllImages(); }
}

async function openFile(imagePath){
  const r = await fetch('/open_file', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({path: imagePath})});
  const d = await r.json();
  if(!d.ok){ showStatus('copystatus', 'Could not open file: ' + (d.error||''), true); }
}

async function revealFile(imagePath){
  const r = await fetch('/reveal_file', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({path: imagePath})});
  const d = await r.json();
  if(!d.ok){ showStatus('copystatus', 'Could not open folder: ' + (d.error||''), true); }
}

async function browse(fieldId){
  const r = await fetch('/browse_folder');
  const d = await r.json();
  if(d.ok && d.path){
    document.getElementById(fieldId).value = d.path;
  } else if(!d.ok){
    showStatus('folderstatus', 'Folder picker unavailable - type the path instead.', true);
  }
}

function showStatus(elId, text, isError){
  const el = document.getElementById(elId);
  el.textContent = text;
  el.style.color = isError ? '#f87171' : '#4ade80';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.textContent = ''; }, 4000);
}

async function saveFolders(){
  const body = {
    sources: getSourcePaths(), thumbs: f_thumbs.value,
    destination: f_dest.value
  };
  const r = await fetch('/set_folders', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const d = await r.json();
  await refreshSourceDropdown();
  showStatus('folderstatus', '\u2713 Folders saved.', false);
}

async function startIndex(){
  await saveFolders();
  showStatus('folderstatus', 'Processing...', false);
  document.getElementById('progtext').textContent = 'Starting indexing\u2026';
  document.getElementById('cancelbtn').style.display = '';
  await fetch('/index', {method:'POST'});
  pollProgress();
}

async function cancelIndex(){
  await fetch('/cancel_index', {method:'POST'});
}

async function pollProgress(){
  const r = await fetch('/progress'); const p = await r.json();
  const pct = p.total ? Math.round(100*p.done/p.total) : (p.running?0:0);
  let line = p.message;
  if(p.running && p.total){
    line = p.message + '  \u2014  ' + p.done + ' / ' + p.total + ' (' + pct + '%)';
  }
  document.getElementById('progtext').textContent = line;
  barfill.style.width = (p.total ? pct : (p.running?4:0)) + '%';

  if (p.running){
    setTimeout(pollProgress, 500);
  } else {
    document.getElementById('cancelbtn').style.display = 'none';
    refreshSourceDropdown();
    loadAllImages();
  }
}

function applySort(){
  const mode = (document.getElementById('sortmode') || {}).value || 'relevance';
  const byName = (a,b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase());
  const px = it => (it.width||0) * (it.height||0);   // total pixels
  const byRes = (a,b) => px(a) - px(b);
  if(mode === 'name')       results.sort(byName);
  else if(mode === 'name_desc') results.sort((a,b)=>byName(b,a));
  else if(mode === 'res')   results.sort(byRes);
  else if(mode === 'res_desc') results.sort((a,b)=>byRes(b,a));
  else results.sort((a,b)=>(a._ord||0)-(b._ord||0));   // relevance / arrival order
}

async function runSearch(keepQuery){
  const q = keepQuery ? lastQuery : searchbar.value.trim();
  if(!q){ lastQuery = ""; loadAllImages(); return; }
  lastQuery = q;
  const r = await fetch('/search?q=' + encodeURIComponent(q)
                        + '&source=' + encodeURIComponent(currentScope));
  results = await r.json();
  results.forEach((it,i) => it._ord = i);
  applySort();
  page = 0;
  renderPage();
}

function perPage(){ return parseInt(document.getElementById('perpage').value); }
function pageCount(){ return Math.max(1, Math.ceil(results.length / perPage())); }

let copiedSet = new Set();
let showCopied = true;

function toggleShowCopied(){
  showCopied = !showCopied;
  const btn = document.getElementById('showCopiedBtn');
  btn.textContent = showCopied ? 'Disable Show Copied' : 'Show Copied';
  btn.classList.toggle('active', showCopied);
  renderPage();
}

function renderPage(){
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const start = page * perPage();
  const slice = results.slice(start, start + perPage());

  for(const item of slice){
    const isCopied = showCopied && (copiedSet.has(item.image) || item.copied);
    const tile = document.createElement('div');
    tile.className = 'tile'
      + (selected.has(item.image) ? ' selected':'')
      + (isCopied ? ' copied':'');
    if(!isCopied){
      tile.onclick = () => toggleSelect(item.image, tile);
    }

    const img = document.createElement('img');
    img.src = '/thumb?path=' + encodeURIComponent(item.thumb);
    img.loading = 'lazy';   // only load thumbnails as they scroll into view
    tile.appendChild(img);

    const meta = document.createElement('div');
    meta.className = 'meta';

    const nameLine = document.createElement('div');
    nameLine.className = 'namerow';
    nameLine.textContent = item.name;
    meta.appendChild(nameLine);

    const infoRow = document.createElement('div');
    infoRow.className = 'inforow';
    const likenessHtml = (item.score !== null && item.score !== undefined)
      ? '<span class="pill likeness">likeness: ' + item.score + '</span>' : '';
    const resHtml = item.resolution
      ? '<span class="pill res">' + item.resolution + '</span>' : '';
    const mpHtml = item.megapixels
      ? '<span class="pill res">' + item.megapixels + ' MP</span>' : '';
    infoRow.innerHTML = likenessHtml + resHtml + mpHtml;

    const openBtn = document.createElement('button');
    openBtn.className = 'openbtn';
    openBtn.textContent = 'Open File';
    openBtn.onclick = (e) => { e.stopPropagation(); openFile(item.image); };

    const revealBtn = document.createElement('button');
    revealBtn.className = 'openbtn';
    revealBtn.textContent = 'Open in Folder';
    revealBtn.onclick = (e) => { e.stopPropagation(); revealFile(item.image); };

    const btnGroup = document.createElement('div');
    btnGroup.className = 'btngroup';
    btnGroup.appendChild(openBtn);
    btnGroup.appendChild(revealBtn);

    meta.appendChild(infoRow);
    meta.appendChild(btnGroup);
    tile.appendChild(meta);

    const check = document.createElement('div');
    check.className = 'check'; check.textContent = '\u2713';
    tile.appendChild(check);

    if(isCopied){
      const tag = document.createElement('div');
      tag.className = 'copiedtag'; tag.textContent = 'copied';
      tile.appendChild(tag);
    }

    grid.appendChild(tile);
  }
  const curPage = results.length ? (page+1) : 0;
  const totPages = results.length ? pageCount() : 0;
  const n = results.length;
  const totalLabel = (n === 1 ? '1 image' : n + ' images');
  document.getElementById('pagenum').value = curPage;
  document.getElementById('pagenum_b').value = curPage;
  document.getElementById('pageinfo').textContent = '/ ' + totPages;
  document.getElementById('pageinfo_b').textContent = '/ ' + totPages;
  document.getElementById('totalinfo').textContent = totalLabel;
  document.getElementById('totalinfo_b').textContent = totalLabel;
  document.getElementById('bottomnav').style.display = (totPages > 1) ? '' : 'none';
  updateCount();
}

function toggleSelect(image, tile){
  if(selected.has(image)){ selected.delete(image); tile.classList.remove('selected'); }
  else { selected.add(image); tile.classList.add('selected'); }
  updateCount();
}
function updateCount(){ document.getElementById('count').textContent = selected.size + ' selected'; }
function selectAllOnPage(){
  const start = page*perPage();
  results.slice(start, start+perPage())
    .filter(i => !(showCopied && (copiedSet.has(i.image) || i.copied)))
    .forEach(i => selected.add(i.image));
  renderPage();
}
function clearSelection(){ selected.clear(); renderPage(); }

function scrollToGridTop(){
  const grid = document.getElementById('grid');
  const y = grid.getBoundingClientRect().top + window.scrollY - 12;
  window.scrollTo({ top: Math.max(0, y), behavior: 'smooth' });
}
function nextPage(){
  if(page < pageCount()-1){ page++; selected.clear(); renderPage(); scrollToGridTop(); }
}
function prevPage(){
  if(page > 0){ page--; selected.clear(); renderPage(); scrollToGridTop(); }
}
function gotoPageFrom(boxId){
  let n = parseInt(document.getElementById(boxId).value);
  if(isNaN(n)) return;
  n = Math.max(1, Math.min(pageCount(), n));
  page = n - 1;
  selected.clear();
  renderPage();
  scrollToGridTop();
}
function gotoPage(){ gotoPageFrom('pagenum'); }
function gotoPageBottom(){ gotoPageFrom('pagenum_b'); }

async function copySelected(){
  if(selected.size === 0){
    showStatus('copystatus', 'Nothing selected.', true);
    return;
  }
  const r = await fetch('/copy', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({images: Array.from(selected)})});
  const d = await r.json();
  (d.copied_images || []).forEach(v => { copiedSet.add(v); selected.delete(v); });
  renderPage();
  let msg = '\u2713 Copied ' + d.copied + ' file(s).';
  if(d.skipped) msg += ' Skipped ' + d.skipped + '.';
  showStatus('copystatus', msg, false);
}
</script>
</body>
</html>
"""


# =============================================================================
#  STARTUP
# =============================================================================

def open_browser_later():
    """Wait a moment for the server to be ready, then open the app in the
    default browser. Runs in a background thread."""
    time.sleep(1.2)
    try:
        webbrowser.open(f"http://localhost:{APP_PORT}")
    except Exception:
        pass


if __name__ == "__main__":
    load_settings()   # pre-fill the folder boxes with last-used paths
    if STATE["folders"].get("thumbs"):
        load_index()  # try loading an existing index so search works immediately
    threading.Thread(target=open_browser_later, daemon=True).start()
    print("[ImageFinder] Starting\u2026  open http://localhost:%d if it "
          "doesn't open on its own." % APP_PORT)
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT, log_level="warning")
