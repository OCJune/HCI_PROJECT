import shutil
import sys
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline import DEFAULT_GENERATED_DIR, generate_coloring_book


UPLOAD_DIR = BASE_DIR / "uploads"
GENERATED_DIR = DEFAULT_GENERATED_DIR

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Coloring Book Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")


@app.get("/api/health")
def health():
    return {"status": "ok"}


DIFFICULTY_TO_K = {
    "쉬움": 10,
    "보통": 20,
    "어려움": 30,
    "easy": 10,
    "normal": 20,
    "hard": 30,
}

DOWNLOAD_ASSETS = {
    "coloring": {
        "filename": "final_numbered_coloringbook.png",
        "download_name": "coloring-book.png",
    },
    "combined": {
        "filename": "numbered_with_color_index.png",
        "download_name": "coloring-book-with-palette.png",
    },
    "palette": {
        "filename": "palette.png",
        "download_name": "coloring-book-palette.png",
    },
}


def resolve_color_count(k, difficulty):
    if k is not None:
        return k
    if difficulty not in DIFFICULTY_TO_K:
        raise HTTPException(status_code=400, detail=f"unsupported difficulty: {difficulty}")
    return DIFFICULTY_TO_K[difficulty]


@app.post("/api/generate")
def generate(
    image: UploadFile = File(...),
    k: int | None = Form(None),
    difficulty: str = Form("보통"),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image must be an image file")

    color_count = resolve_color_count(k, difficulty)
    result_id = uuid4().hex
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".png"

    upload_path = UPLOAD_DIR / f"{result_id}{suffix}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        result = generate_coloring_book(
            upload_path,
            k=color_count,
            output_root=GENERATED_DIR,
            result_id=result_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def url_for_asset(name):
        return f"/generated/{result_id}/{name}"

    def download_url(asset_name):
        return f"/api/results/{result_id}/download/{asset_name}"

    return {
        "result_id": result["result_id"],
        "original_url": url_for_asset("original.png"),
        "coloring_url": url_for_asset("final_numbered_coloringbook.png"),
        "palette_url": url_for_asset("palette.png"),
        "combined_url": url_for_asset("numbered_with_color_index.png"),
        "download_url": download_url("coloring"),
        "download_urls": {
            "coloring": download_url("coloring"),
            "combined": download_url("combined"),
            "palette": download_url("palette"),
        },
        "preview_urls": {
            "kmeans": url_for_asset("kmeans.png"),
            "segmentation": url_for_asset("region_preview.png"),
            "colored_by_labels": url_for_asset("colored_by_labels_numbered.png"),
        },
        "palette": result["palette"],
        "metrics": result["metrics"],
        "image_size": result["image_size"],
        "difficulty": difficulty,
        "k": color_count,
    }


@app.get("/api/results/{result_id}/download/{asset_name}")
def download_result(result_id: str, asset_name: str):
    if asset_name not in DOWNLOAD_ASSETS:
        raise HTTPException(status_code=404, detail="download asset not found")

    if not result_id.isalnum():
        raise HTTPException(status_code=400, detail="invalid result id")

    asset = DOWNLOAD_ASSETS[asset_name]
    file_path = GENERATED_DIR / result_id / asset["filename"]
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="result file not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=asset["download_name"],
    )
