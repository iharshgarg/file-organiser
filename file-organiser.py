from pathlib import Path
import shutil
import time
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from google import genai
from config import API_KEY

# ---------------- CONFIG ----------------
MODEL = "models/gemini-flash-latest"

BASE_FOLDER = Path.home() / "Projects" / "Pybox"

IGNORED_FILENAMES = {
    ".DS_Store",
    ".gitignore",
    "Thumbs.db",
}

DEBUG = True
MAX_FOLDERS = 5   # 🔴 HARD LIMIT
# ---------------------------------------

client = genai.Client(api_key=API_KEY)


def log(msg: str):
    if DEBUG:
        print(msg)


def should_ignore(file_path: Path) -> bool:
    return (
        file_path.name.startswith(".")
        or file_path.name in IGNORED_FILENAMES
    )


def sanitize_folder(name: str) -> str:
    return name.replace("/", "_").strip()


# ---------- Batch classification (CRITICAL PART) ----------

def batch_classify(files: list[Path]) -> dict[str, list[Path]]:
    log("\n🧠 Starting SIMILARITY-FIRST batch classification")
    log(f"📄 Files: {[f.name for f in files]}")

    file_list = [f.name for f in files]

    prompt = f"""
You are a PERSONAL file organization assistant.

PRIMARY GOAL:
Group files by SIMILARITY, using AS FEW FOLDERS AS POSSIBLE.

HARD CONSTRAINTS (DO NOT VIOLATE):
- Create AT MOST {MAX_FOLDERS} folders total
- Prefer merging files over creating new folders
- If two files are even loosely related, put them together
- Folder creation is EXPENSIVE — avoid it
- First cluster files by similarity, THEN name the clusters

ORGANIZATION STYLE:
- Think like a human cleaning their personal laptop
- Group by life context or purpose
- Broad, reusable folders only

GOOD folder names:
2019 School trip, college projects, marksheets, reciepts, songs

BAD folder names(too vague or too specific):
photo-3675472, files, documents,..(try to avoid generic names)

Return ONLY valid JSON.
No markdown. No explanations.

Format:
{{
  "FolderName": ["file1.ext", "file2.ext", "..."]
}}

Files:
{file_list}
"""

    log("\n📤 Prompt sent to LLM:")
    log(prompt)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    raw = response.text.strip()
    log("\n📥 RAW LLM RESPONSE:")
    log(raw)

    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except Exception:
        print("❌ JSON parse failed")
        print(raw)
        return {}

    folder_map: dict[str, list[Path]] = {}

    for folder, filenames in data.items():
        clean_folder = sanitize_folder(folder)
        for f in files:
            if f.name in filenames:
                folder_map.setdefault(clean_folder, []).append(f)

    log("\n📂 Final clusters:")
    for folder, files in folder_map.items():
        log(f"  {folder}: {[f.name for f in files]}")

    return folder_map


def apply_moves(folder_map: dict[str, list[Path]]):
    log("\n🚚 Applying moves")
    for folder, files in folder_map.items():
        target = BASE_FOLDER / folder
        target.mkdir(exist_ok=True)

        for f in files:
            try:
                shutil.move(str(f), target / f.name)
                print(f"📁 {f.name} → {folder}/")
            except Exception as e:
                print(f"❌ Move failed for {f.name}: {e}")


# ---------- Initial batch scan ----------

print("🔍 Initial batch scan...")

files_to_process = [
    f for f in BASE_FOLDER.iterdir()
    if f.is_file() and not should_ignore(f)
]

log(f"📄 Found {len(files_to_process)} files")

if files_to_process:
    folder_map = batch_classify(files_to_process)
    apply_moves(folder_map)


# ---------- Live mode (REUSE ONLY) ----------

def classify_single_file(file_path: Path) -> str | None:
    existing_folders = [
        f.name for f in BASE_FOLDER.iterdir()
        if f.is_dir()
    ]

    prompt = f"""
You are organizing a new file.

IMPORTANT:
Try to prefer using existing folders.
But in some cases can create new folders too.

Existing folders:
{existing_folders}

Choose the BEST match based on similarity and purpose.

Return JSON only.

Format:
{{ "folder": "ExistingFolderName" }}

File name: {file_path.name}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
        return sanitize_folder(data.get("folder", ""))
    except Exception:
        return None


def handle_new_file(file_path: Path):
    if not file_path.is_file():
        return
    if file_path.parent != BASE_FOLDER:
        return
    if should_ignore(file_path):
        return

    folder = classify_single_file(file_path)
    if not folder:
        return

    target = BASE_FOLDER / folder
    if not target.exists():
        return  # safety: never create new folders in live mode

    try:
        shutil.move(str(file_path), target / file_path.name)
        print(f"📁 {file_path.name} → {folder}/")
    except Exception:
        pass


class OrganizerHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        time.sleep(0.5)
        handle_new_file(Path(event.src_path))


observer = Observer()
observer.schedule(OrganizerHandler(), str(BASE_FOLDER), recursive=False)
observer.start()

print("👀 AI organizer running (similarity-first)... Ctrl+C to stop")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
