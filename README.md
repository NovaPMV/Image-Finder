# Image Finder — by NovaPMV

A local app that lets you **search a folder of images by typing what you want**
("girl in a red dress", "wearing glasses", "tongue out") and instantly
get a grid of matching pictures. Select the ones you like and copy the originals
into a folder, ready to use however you need. You can also just browse everything
and copy without searching.

Supports **multiple source folders at once** — point it at several picture
folders and search across all of them together, or filter down to just one. It
also scans **subfolders** automatically, so you can point it at one big parent
folder and it finds everything inside.

It runs **entirely on your own computer**. Once you've used it once and the model
is downloaded, it can run offline. If you have an NVIDIA graphics card it uses it
automatically for speed; otherwise it runs on your CPU (slower, but works fine).

**Unlike the video version, there's no ffmpeg and no frames** — an image is
already a single picture, so setup is simpler and indexing is much faster and
lighter on disk.

---

## What it does (the short version)

1. You pick one or more **source folders** of images.
2. The app makes a tiny **thumbnail** of each picture for the grid, and learns
   what each one looks like.
3. You type a description and it ranks your pictures by how well they match.
4. You tick the ones you want and copy the **originals** to a destination folder.

That's the whole thing. If you've used TikTok Sorter, this is the same idea with
the video-specific parts removed.

---

## What you need (one-time setup)

You'll install two things once: **Python** and some **Python packages**. There's
**no ffmpeg to install** — that was only needed for video. Follow every step in
order. It looks long because it's spelled out for total beginners — the actual
work is maybe 5–10 minutes plus download time.

### Step 1 — Install Python

1. Go to https://www.python.org/downloads/ and download Python (3.10 or newer).
2. Run the installer.
3. **VERY IMPORTANT:** on the first screen, check the box that says
   **"Add Python to PATH"** before clicking Install. If you miss this, nothing
   else will work. If you're not sure whether you did, just reinstall and tick it.

To confirm it worked: open **Command Prompt** (press the Windows key, type `cmd`,
hit Enter) and type `python --version`. If you see a version number, you're good.

### Step 2 — Install the Python packages

In **Command Prompt**, run these **one line at a time**. Wait for each to finish
before running the next. Doing them one by one avoids errors.

    pip install numpy
    pip install pillow
    pip install pillow-heif
    pip install fastapi
    pip install "uvicorn[standard]"
    pip install open_clip_torch

(`pillow-heif` lets the app open iPhone `.heic` photos. It's optional but handy —
if you never use HEIC files you can skip it.)

Now install **torch** (the AI engine). **Pick the ONE that matches your machine:**

**Option A — you HAVE an NVIDIA graphics card (recommended, much faster):**

    pip install torch --index-url https://download.pytorch.org/whl/cu121

**Option B — you do NOT have an NVIDIA card (or Option A fails):**

    pip install torch

That's it for setup.

### Step 3 — Get the app files

Download this repository (green **Code** button → **Download ZIP**), and unzip it
somewhere easy like your Desktop. You should have `imagefinder.py`, the launcher,
and this README together in one folder. **Keep them together.**

---

## Running it

Double-click **`Start ImageFinder.bat`**.

- A black Command Prompt window opens. **Leave it open.** That window *is* the
  program. It's not an error — it's the engine running.
- Your browser opens to the app automatically. If it doesn't, open your browser
  and go to **http://localhost:8001** yourself.

**When you're finished, close that window to shut the app down.** Reopen the
`.bat` next time. That's the whole on/off switch.

> If you'd rather not use the launcher, you can also start it from Command Prompt
> with `python imagefinder.py` from inside the app folder.

### First launch is slow — this is normal

The **very first time** you build an index, the app downloads the AI model
(about **1.5 GB**, one time — it's cached forever after). Then it looks at every
image, which takes a while. ***I recommend choosing a folder with under a few
hundred images for your first index.*** A folder of several thousand pictures can
take **many minutes**, especially on CPU. **Let it run.** The progress bar shows a
live count (e.g. "370 / 2000"). You can hit **Cancel** anytime — it keeps
everything already done, and next time it only processes what's left.

**Updates after that are fast** — it only processes new or changed images when you
manually update the index. Images are quicker to index than videos, since there
are no frames to extract.

---

## How to use it

### 1. Set your folders

Each folder has a **Browse** button (or you can paste the path):

- **Source folders** — where your images are. **You can add more than one.**
  Click **+ Add source folder** for each folder you want included. Each row has
  its own Browse and Remove button. They all get indexed together into one
  searchable library, **including all subfolders**.
- **Thumbnails folder** — an empty folder the app fills with tiny copies of your
  images for the grid. You never open this; it's behind-the-scenes. **One shared
  Thumbnails folder holds everything, even with multiple sources** — images can't
  collide because each is stored under a unique ID.
- **Destination folder** — where copies of the images you pick get sent.

Then click **Save folders**. The app remembers all of these next time — including
your whole list of source folders — so you only redo this when you want to change
something.

> **Heads-up:** the Browse folder-picker sometimes opens *behind* your browser
> window. If you click Browse and nothing appears, **Alt+Tab** to find it. Or just
> paste the folder path into the box.

### 2. Build the index

Click **Build / update index** and wait (see "first launch is slow" above). It
walks every source folder you added, including subfolders. Adding a new folder
later only indexes that folder's new images — it doesn't redo the old ones.

### 3. Search or browse

Type a phrase and press Enter to search — or just browse everything that's already
indexed without searching. (Browsing works even before you search, since the
thumbnails are ready as soon as indexing finishes.)

**All folders vs. Single folder:** use the tabs above the grid.

- **All folders** searches and browses across every source at once.
- **Single folder** shows a dropdown — pick one folder and the grid narrows to
  just that one. Switching is instant and never re-indexes. Great for when you
  only want to pull from one place.

**Sort by** lets you order the grid by relevance (best matches first, for
searches), filename (A–Z or Z–A), or resolution (highest or lowest first).

### 4. Select the images you want

**Click an image to select it** (it gets a bright blue glowing border). Click
again to deselect. Use **Select all on page** or **Clear selection** for bulk
picking. The running count shows how many you've selected.

By default, images you've already copied this session grey out with a "copied" tag
so you don't copy duplicates. If you actually *want* to copy something again, click
**Disable Show Copied** — that turns off the grey overlay and lets you re-select
copied images. Click it again ("Show Copied") to turn the greying back on.

### 5. Copy or open

- **Copy selected to destination** copies the original files into your destination
  folder, ready to use.
- **Open File** on any image opens the original in your normal image viewer.
- **Open in Folder** opens the image's folder with the file already highlighted —
  the easiest way to drag it straight into another app.

> **Note:** moving to another page clears your current selection — so copy the ones
> you want from a page *before* changing pages. Page navigation is available at both
> the top and bottom of the grid.

### What the tags under each image mean

- **likeness** — how well the image matched your search (higher = better match).
  Only shows when you've searched. Treat it as a ranking guide, not an exact
  score — the numbers are relative, so the same value can mean different things
  for different searches.
- **resolution** — the original image's pixel size, e.g. `4032×3024`.
- **MP** — megapixels (roughly, how many millions of pixels), a quick sense of how
  large/detailed the image is.

---

## How much disk space does it use?

The app makes one small **thumbnail** per image. These are tiny compared to your
originals. Each thumbnail is a JPEG whose longest side is 400 pixels — usually
**~20–60 KB each**. So:

| How many images | Rough thumbnail space |
|---|---|
| 1,000 images | ~**20–60 MB** |
| 10,000 images | ~**200–600 MB** |

Plus a small index file that remembers what each image looks like. This is
**much smaller than the video version**, since there are no frames or preview
clips. Your original images are never touched — the app only *reads* and *copies*
them. If space gets tight, see the tuning options below.

---

## Optional Tuning — Making it faster or more accurate

All of these are simple edits near the **top of `imagefinder.py`**. Open it in any
text editor (Notepad works). Change a value, save, and **rebuild the index** for
it to take effect.

> **Whenever you change the model, do a clean rebuild:** delete
> `imagefinder_index.json` and `imagefinder_vectors.npy` from your **Thumbnails
> folder** (and you can clear the old thumbnails too), then click Build / update
> index. Otherwise old data mixes with new and your counts look wrong.

### Faster indexing — use a lighter model

Find these lines near the top:

    CLIP_MODEL_NAME = "ViT-L-14"
    CLIP_PRETRAINED = "laion2b_s32b_b82k"
    # CLIP_MODEL_NAME = "ViT-B-32"
    # CLIP_PRETRAINED = "laion2b_s34b_b79k"

The app ships with **ViT-L-14** (most accurate, but slower to index). For faster
indexing at slightly lower search precision, **comment out the two L-14 lines and
uncomment the two B-32 lines** (swap which pair has the `#` in front):

    # CLIP_MODEL_NAME = "ViT-L-14"
    # CLIP_PRETRAINED = "laion2b_s32b_b82k"
    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "laion2b_s34b_b79k"

Rough guide to your options (all run locally, all free):
- **ViT-B-32** — fastest, smallest download (~600 MB), good for most searches.
- **ViT-L-14** — the default. Slower, ~1.5 GB, noticeably more precise.
- **ViT-H-14** (`laion2b_s32b_b79k`) — even more accurate, but big and slow;
  only worth it on a strong GPU with lots of VRAM.

If you switch to a model not listed here, you may also need to update the
`MODEL_VECTOR_DIM` line (768 for L-14/H-14, 512 for B-32) — the file has a comment
explaining it.

### Faster indexing — batch size

Find:

    INDEX_BATCH_SIZE = 16

This controls **how many images the app looks at in one go** while indexing.
Instead of processing pictures one at a time, it hands the AI a whole stack at
once, which is a lot more efficient — this is the single biggest speed setting for
indexing, especially on CPU.

**What the number does:** bigger = faster indexing, but it uses more memory (RAM
on CPU, or video memory / VRAM on an NVIDIA GPU) while it runs. Smaller = slower,
but lighter on memory. It has **no effect on search speed or accuracy** — it only
changes how quickly the index gets built. Your results are identical either way.

**When to raise it:**
- You're on a **strong GPU with lots of VRAM** (8 GB+) — try `32`, or `64` if
  indexing still isn't maxing out your card. Larger batches keep a good GPU busier.
- You have **lots of system RAM** and indexing feels slow on CPU — bumping to `32`
  can help a bit, though CPU gains taper off faster than GPU gains.

**When to lower it:**
- You get an **"out of memory" error** while indexing (it may say "CUDA out of
  memory" on a GPU) — drop it to `8`, or `4` if that still isn't enough. This is
  the main reason to change it.
- You're on a **6 GB GPU** (like a 1660 Super) and using the big **ViT-L-14**
  model — `16` is a safe default, but `8` gives you more headroom if it's tight.
- Your machine has **limited RAM** and feels sluggish while indexing.

`16` is a sensible middle-ground that works on most setups. If you're not hitting
memory errors and indexing speed is fine, there's no need to touch it. Unlike the
model or thumbnail settings, changing this does **not** require a clean rebuild —
it only affects speed, not the data that gets stored, so you can change it anytime.

### Bigger or smaller thumbnails

Find:

    THUMB_MAX = 400

This is the longest side of each thumbnail, in pixels. Lower it (e.g. `300` or
`250`) for smaller files and less disk use. Raise it (e.g. `500`) for sharper grid
tiles that take more space. `400` is a good balance.

### Change how many images show per page

The grid shows **24 images per page** by default. You can change this live with the
**Per page** slider in the app (no code editing needed). If you want a different
starting default, find this line near the top of `imagefinder.py`:

    DEFAULT_PER_PAGE = 24

### Which file types it includes

Find:

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif",
                        ".bmp", ".tiff", ".tif", ".heic", ".heif"}

Add or remove extensions here if you want to include or ignore certain formats.
(`.heic`/`.heif` only work if you installed `pillow-heif` in setup.)

---

## Switching between GPU and CPU

The app uses your NVIDIA GPU automatically **if** you installed the GPU version of
torch. **How to check which you have:** look at the black Command Prompt window
when the app starts. It prints one of:

    Running on GPU (CUDA): NVIDIA GeForce GTX 1660 SUPER
    Running on CPU (no supported GPU detected).

**To switch from CPU to GPU** (you have an NVIDIA card and want the speed):

    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu121

**To switch from GPU back to CPU:**

    pip uninstall -y torch
    pip install torch

Restart the app after switching. No code changes needed — it detects the right one
on its own.

---

## Uninstalling / cleaning up

There's no installer, so there's nothing to uninstall from your system settings —
everything lives in plain folders you can delete by hand. Here's every place the
app leaves something.

### 1. The app files themselves

Wherever you unzipped the download (e.g. your Desktop). Delete the whole folder —
it contains `imagefinder.py`, the launcher, and this README. Deleting these stops
the app from running; it does **not** touch your images or the AI model cache.

### 2. The AI model cache (~1.5 GB — the big one)

The CLIP model downloads once and is cached here:

    C:\Users\<your-name>\.cache\huggingface\

Replace `<your-name>` with your Windows username. This is the largest thing the app
leaves behind. Deleting it frees up the most space; the only cost is that the model
will re-download (~1.5 GB) if you ever run the app again. To reach a hidden `.cache`
folder, paste the path straight into the File Explorer address bar.

### 3. Your settings file

Sits next to `imagefinder.py` in the app folder:

    imagefinder_settings.json

This just remembers your folder paths. Deleting it is harmless — the app simply
forgets those paths and you re-pick them next time.

### 4. The index (the app's "memory" of your images)

Stored inside whatever you chose as your **Thumbnails folder**:

    imagefinder_index.json
    imagefinder_vectors.npy

These are what make search instant without re-scanning. Delete them if you want the
app to forget everything it indexed (you'd rebuild the index next time).

### 5. The thumbnails

Also in your **Thumbnails folder** — the tiny image copies the app generated. Safe
to delete; they only get regenerated when you re-index.

### 6. The Python packages (optional)

If you don't use Python for anything else and want the space back:

    pip uninstall -y torch open_clip_torch fastapi "uvicorn[standard]" pillow pillow-heif numpy

Only do this if you're sure nothing else on your machine needs them. **torch is the
big one here** (often several GB for the GPU build), so uninstalling it reclaims a
lot of space.

> **The short version:** to reclaim the most space fast, delete the app folder and
> the `huggingface` cache in step 2. Your original images are never touched by any
> of this — the app only ever *reads* and *copies* them, never moves or deletes.

---

## Common problems (read this before asking)

- **"python is not recognized"** → Python isn't on your PATH. Reinstall Python and
  tick **"Add Python to PATH"** on the first screen.
- **A package failed to install** → Install them one line at a time (Step 2), not
  all in one line.
- **The app page won't load / "can't connect"** → Give it a few seconds after
  launch and refresh. The server takes a moment to start on the first run.
- **It says "Running on CPU" but I have an NVIDIA card** → You installed the CPU
  torch. Use the "switch from CPU to GPU" commands above.
- **`.heic` iPhone photos don't show up** → Install the HEIC plugin:
  `pip install pillow-heif`, then rebuild the index.
- **The Browse button does nothing** → The folder picker opened behind your
  browser. **Alt+Tab** to find it. Or just paste the folder path into the box.
- **The Single folder dropdown is empty** → A folder shows up there once it's been
  saved or indexed. Add your sources, click Save folders, and build the index.
- **The indexed count looks wrong (too high)** → Leftover data from an earlier run
  with different settings. Do a clean rebuild (delete `imagefinder_index.json` and
  `imagefinder_vectors.npy` from your Thumbnails folder, then reindex).
- **The black window closed and the app stopped** → That window is the program.
  Keep it open while using the app; reopen the `.bat` to start again.
- **Windows SmartScreen or antivirus warns about the `.bat`** → It's a plain text
  launcher you can open in Notepad to inspect. Allow it if you trust the source.

---

## A note from Nova

This is super similar to my TikTok Sorter project — exact same layout, but for
pictures instead of clips, and simpler under the hood since there's no ffmpeg or
frames involved. Since I reused the same UI, it's not as optimized for displaying 
images, but it still works just fine. Feel free to mess around with the code if you 
want to make something that works a bit better. If you have any issues or
suggestions, message me on Discord @novapmv. Hope you get some use out of it!
