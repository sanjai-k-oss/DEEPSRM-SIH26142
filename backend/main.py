import os
import gc
import io
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from PIL import Image
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="DEEPSRM API", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

jobs = {}

TEMP_DIR = BASE_DIR / "temp_results"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def save_temp_result(job_id: str, name: str, data: bytes) -> str:
    path = TEMP_DIR / f"{job_id}_{name}"
    path.write_bytes(data)
    return str(path)

def delete_temp_file(path):
    try:
        if path:
            Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def cleanup_old_temp_files(max_age_seconds=1800):
    """Delete temporary result files older than 30 minutes."""
    import time

    now = time.time()

    try:
        for path in TEMP_DIR.iterdir():
            if not path.is_file():
                continue

            try:
                age = now - path.stat().st_mtime

                if age > max_age_seconds:
                    path.unlink(missing_ok=True)

            except Exception:
                pass

    except Exception:
        pass

# Super-resolution model selection. Render's 512 MB instance is too small for
# the EDSR graph, so the default deployment model is FSRCNN-small x4.
# Set DEEPSRM_SR_MODEL=edsr to use the existing EDSR model locally.
SR_MODEL_NAME = os.getenv("DEEPSRM_SR_MODEL", "fsrcnn-small").lower().strip()

SR_MODELS = {
    "edsr": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "path": BASE_DIR / "models" / "EDSR_x4.pb",
        "cv_name": "edsr",
    },
    "fsrcnn": {
        "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
        "path": BASE_DIR / "models" / "FSRCNN_x4.pb",
        "cv_name": "fsrcnn",
    },
    "fsrcnn-small": {
        "url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN-small_x4.pb",
        "path": BASE_DIR / "models" / "FSRCNN-small_x4.pb",
        "cv_name": "fsrcnn",
    },
}

if SR_MODEL_NAME not in SR_MODELS:
    SR_MODEL_NAME = "fsrcnn-small"


def ensure_sr_model():
    """Download the selected OpenCV dnn_superres model on first use."""
    cfg = SR_MODELS[SR_MODEL_NAME]
    model_path = cfg["path"]
    if model_path.exists() and model_path.stat().st_size > 1_000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(cfg["url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:
                    f.write(chunk)
    return model_path


def ai_rgb_geotiff(raw_tiff: bytes, scale=4):
    """Run a lightweight pretrained neural SR model on the RGB view.

    Render uses FSRCNN-small by default because its model is dramatically
    smaller than EDSR and is designed for fast super-resolution inference.
    EDSR remains available by setting DEEPSRM_SR_MODEL=edsr.
    This is an AI visual-SR demonstration, not the final multispectral
    satellite-trained DEEPSRM model.
    """
    import gc

    model_path = ensure_sr_model()

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(SR_MODELS[SR_MODEL_NAME]["cv_name"], scale)

    with MemoryFile(raw_tiff) as mem:
        with mem.open() as src:
            rgb = src.read([3, 2, 1]).astype(np.float32)
            np.nan_to_num(rgb, copy=False)

            positives = rgb[rgb > 0]
            p98 = np.percentile(positives, 98) if positives.size else 1.0
            del positives

            x = np.clip(rgb / max(p98, 1e-6), 0, 1)
            del rgb

            bgr = (x[::-1].transpose(1, 2, 0) * 255).astype(np.uint8)
            del x
            gc.collect()

            tile_size = 32
            h0, w0 = bgr.shape[:2]
            out_bgr = np.zeros((h0 * scale, w0 * scale, 3), dtype=np.uint8)

            for y in range(0, h0, tile_size):
                for x0 in range(0, w0, tile_size):
                    y1 = min(y + tile_size, h0)
                    x1 = min(x0 + tile_size, w0)
                    tile = bgr[y:y1, x0:x1]
                    tile_out = sr.upsample(tile)

                    oh = (y1 - y) * scale
                    ow = (x1 - x0) * scale
                    out_bgr[y * scale:y * scale + oh, x0 * scale:x0 * scale + ow] = tile_out[:oh, :ow]

                    del tile, tile_out

            del bgr, sr
            gc.collect()

            out_rgb = out_bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.uint8)
            del out_bgr
            gc.collect()

            h, w = out_rgb.shape[1:]
            profile = src.profile.copy()
            profile.update(
                count=3,
                dtype="uint8",
                height=h,
                width=w,
                transform=src.transform * src.transform.scale(1 / scale, 1 / scale),
                compress="deflate",
            )

            result = io.BytesIO()
            with rasterio.open(result, "w", **profile) as dst:
                dst.write(out_rgb)

            output = result.getvalue()
            del out_rgb, result
            gc.collect()
            return output
def source_preview(raw_tiff: bytes):
    return make_preview_from_geotiff(raw_tiff)


class SentinelSearch(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    max_cloud: float = 30.0
    limit: int = 12

class SentinelProcessRequest(BaseModel):
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    max_cloud: float = 30.0
    size: int = 128

def get_token():
    """Get a Copernicus Data Space OAuth2 client-credentials token."""
    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE credentials are missing. Set CDSE_CLIENT_ID and "
            "CDSE_CLIENT_SECRET in backend/.env."
        )

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }

    response = requests.post(TOKEN_URL, data=data, timeout=30)

    if not response.ok:
        raise RuntimeError(
            f"Copernicus authentication failed ({response.status_code}): "
            f"{response.text[:1000]}"
        )

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError("Copernicus authentication response had no access_token.")

    return token


def point_geometry(lat, lon, delta=0.02):
    # Small AOI around the selected point.
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon-delta, lat-delta],
            [lon+delta, lat-delta],
            [lon+delta, lat+delta],
            [lon-delta, lat+delta],
            [lon-delta, lat-delta],
        ]]
    }

def stac_search(req: SentinelSearch):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{req.start_date}T00:00:00Z/{req.end_date}T23:59:59Z",
        "limit": min(max(req.limit, 1), 12),
        "intersects": point_geometry(req.latitude, req.longitude),
        "query": {
            "eo:cloud_cover": {
                "lte": req.max_cloud
            }
        },
        "sortby": [
            {
                "field": "properties.datetime",
                "direction": "desc"
            }
        ]
    }

    headers = {
        "Accept": "application/geo+json",
        "Content-Type": "application/json",
        "User-Agent": "DEEPSRM-SIH26142/1.0"
    }

    r = requests.post(
        STAC_URL,
        json=payload,
        headers=headers,
        timeout=60
    )

    if not r.ok:
        raise RuntimeError(
            f"Copernicus STAC search error {r.status_code}: "
            f"{r.text[:2000]}"
        )

    return r.json()

def make_evalscript():
    return """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "SCL"],
    output: {
      id: "default",
      bands: 3,
      sampleType: SampleType.UINT8
    }
  };
}

function evaluatePixel(sample) {
  if ([0, 3, 8, 9, 10, 11].includes(sample.SCL)) {
    return [0, 0, 0];
  }

  return [
    Math.min(255, Math.max(0, sample.B04 * 255)),
    Math.min(255, Math.max(0, sample.B03 * 255)),
    Math.min(255, Math.max(0, sample.B02 * 255))
  ];
}
"""


def process_sentinel(req: SentinelProcessRequest):
    token = get_token()

    if not (-90 <= req.latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= req.longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")
    if not (0 <= req.max_cloud <= 100):
        raise ValueError("Cloud cover must be between 0 and 100.")
    if not (64 <= req.size <= 2048):
        raise ValueError("Output size must be between 64 and 2048.")

    bbox = [
        req.longitude - 0.02,
        req.latitude - 0.02,
        req.longitude + 0.02,
        req.latitude + 0.02
    ]

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                }
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{req.start_date}T00:00:00Z",
                        "to": f"{req.end_date}T23:59:59Z"
                    },
                    "maxCloudCoverage": req.max_cloud,
                    "mosaickingOrder": "leastCC"
                
                }
            }]
        },
        "output": {
            "width": req.size,
            "height": req.size,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": make_evalscript()
    }

    r = requests.post(
        PROCESS_URL,
        json=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
            "Content-Type": "application/json"
        },
        timeout=120
    )

    if not r.ok:
        raise RuntimeError(
            f"Copernicus Process API error {r.status_code}: "
            f"{r.text[:3000]}"
        )

    return r.content


def sr_baseline_geotiff(raw_tiff: bytes, scale=4):
    """
    Transparent prototype baseline:
    - reads the Sentinel-derived GeoTIFF
    - upsamples every raster band with Lanczos
    - preserves CRS/transform
    - writes a new GeoTIFF
    This is NOT the trained DEEPSRM neural model.
    Replace this function with SwinIR/ESRGAN inference for the final system.
    """
    with MemoryFile(raw_tiff) as mem:
        with mem.open() as src:
            data = src.read()
            profile = src.profile.copy()
            transform = src.transform

            h, w = data.shape[1:]
            out_h, out_w = h * scale, w * scale
            out = np.zeros((data.shape[0], out_h, out_w), dtype=np.float32)

            for i in range(data.shape[0]):
                # Rasterio's resampling is geospatially aware and avoids
                # changing the band count.
                arr = data[i].astype(np.float32)
                with MemoryFile() as tmp:
                    p = src.profile.copy()
                    p.update(height=h, width=w, count=1, dtype="float32")
                    with tmp.open(**p) as ds:
                        ds.write(arr, 1)
                        out[i] = ds.read(
                            1,
                            out_shape=(out_h, out_w),
                            resampling=Resampling.lanczos
                        )

            profile.update(
                height=out_h,
                width=out_w,
                transform=transform * transform.scale(1/scale, 1/scale),
                dtype="float32",
                count=data.shape[0],
                compress="deflate",
            )

            result = io.BytesIO()
            with rasterio.open(result, "w", **profile) as dst:
                dst.write(out)
            return result.getvalue()

def make_preview_from_geotiff(raw_tiff: bytes):
    with MemoryFile(raw_tiff) as mem:
        with mem.open() as src:
            # B04, B03, B02 are indices 3,2,1 in the 5-band output.
            bands = src.read([3,2,1]).astype(np.float32)
            valid = np.nan_to_num(bands)
            p98 = np.percentile(valid[valid > 0], 98) if np.any(valid > 0) else 1
            rgb = np.clip(valid / max(p98, 1e-6), 0, 1)
            rgb = (rgb.transpose(1,2,0) * 255).astype(np.uint8)
            img = Image.fromarray(rgb, "RGB")
            img.thumbnail((900, 900))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()

@app.get("/api/status")
def status():
    return {
        "status": "online",
        "mode": "real-copernicus-plus-lightweight-sr",
        "sentinel_stac": STAC_URL,
        "process_api": PROCESS_URL,
        "neural_model": f"{SR_MODEL_NAME} x4 RGB visual SR; multispectral baseline preserved"
    }

@app.post("/api/sentinel/search")
def sentinel_search(req: SentinelSearch):
    try:
        data = stac_search(req)
    except Exception as exc:
        raise HTTPException(502, f"Copernicus STAC search failed: {exc}")

    results = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        assets = feature.get("assets", {})
        thumb = None
        for key in ("thumbnail", "rendered_preview", "visual"):
            if key in assets:
                thumb = assets[key].get("href")
                if thumb:
                    break
        results.append({
            "id": feature.get("id"),
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "thumbnail": thumb,
            "geometry": feature.get("geometry"),
        })
    return {"status": "ok", "count": len(results), "features": results}

@app.post("/api/sentinel/process")
def sentinel_process(req: SentinelProcessRequest):
    import gc

    # Remove abandoned temporary results older than 30 minutes.
    cleanup_old_temp_files()

    job_id = str(uuid.uuid4())

    try:
        # Download the Sentinel-2 source only once.
        source = process_sentinel(req)

        # Run EDSR first so the multispectral output is not occupying
        # memory during the neural super-resolution peak.
        ai_rgb = ai_rgb_geotiff(source, scale=4)

        gc.collect()

        # Reuse the same Sentinel source for the geospatial
        # multispectral baseline instead of downloading it again.
        multispectral = sr_baseline_geotiff(source, scale=4)

        # Create the original preview before releasing the source.
        original = source_preview(source)

        del source
        gc.collect()

        enhanced = make_preview_from_geotiff(ai_rgb)

        # Save completed results to disk instead of keeping large
        # TIFF/PNG byte objects in Render RAM.
        geotiff_path = save_temp_result(job_id, "multispectral.tif", multispectral)
        ai_rgb_path = save_temp_result(job_id, "ai_rgb.tif", ai_rgb)
        preview_path = save_temp_result(job_id, "preview.png", enhanced)
        original_preview_path = save_temp_result(
            job_id,
            "original_preview.png",
            original
        )

        # Release large objects immediately.
        del multispectral
        del ai_rgb
        del enhanced
        del original
        gc.collect()

    except Exception as exc:
        gc.collect()
        raise HTTPException(502, f"Sentinel/AI processing failed: {exc}")

    jobs[job_id] = {
        "geotiff_path": geotiff_path,
        "ai_rgb_path": ai_rgb_path,
        "preview_path": preview_path,
        "original_preview_path": original_preview_path,
        "mode": f"{SR_MODEL_NAME}-x4-rgb-plus-multispectral-baseline",
    }

    gc.collect()

    return {
        "id": job_id,
        "mode": f"{SR_MODEL_NAME}-x4-rgb-plus-multispectral-baseline",
        "input_resolution": "Sentinel-2 L2A 10m/20m source bands",
        "target_resolution": "4x output grid (2.5m for 10m bands; 5m for 20m bands)",
        "neural_model": f"{SR_MODEL_NAME} x4 pretrained RGB visual SR",
        "note": "The downloadable multispectral GeoTIFF preserves the Sentinel bands using the geospatial baseline. The AI output is a separate 3-band RGB GeoTIFF. Final DEEPSRM should use a multispectral satellite-trained checkpoint.",
        "download": f"/api/result/{job_id}/download",
        "ai_rgb_download": f"/api/result/{job_id}/ai-rgb-download",
        "preview": f"/api/result/{job_id}/preview",
        "original_preview": f"/api/result/{job_id}/original-preview",
    }
def stream_temp_file(path, media_type, filename, delete_after=False):
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(404, "Result file not found")

    def file_iterator():
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            if delete_after:
                delete_temp_file(file_path)

    return StreamingResponse(
        file_iterator(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.get("/api/result/{job_id}/download")
def download_result(job_id: str):
    job = jobs.get(job_id)

    if not job or "geotiff_path" not in job:
        raise HTTPException(404, "Result not found")

    path = job["geotiff_path"]

    response = stream_temp_file(
        path,
        "image/tiff",
        f"deepsrm_{job_id[:8]}.tif",
        delete_after=True
    )

    # Keep the job metadata so the other result endpoints
    # can still access their files.
    return response


@app.get("/api/result/{job_id}/ai-rgb-download")
def download_ai_rgb(job_id: str):
    job = jobs.get(job_id)

    if not job or "ai_rgb_path" not in job:
        raise HTTPException(404, "AI RGB result not found")

    path = job["ai_rgb_path"]

    response = stream_temp_file(
        path,
        "image/tiff",
        f"deepsrm_edsr_rgb_{job_id[:8]}.tif",
        delete_after=True
    )

    return response


@app.get("/api/result/{job_id}/original-preview")
def original_preview(job_id: str):
    job = jobs.get(job_id)

    if not job or "original_preview_path" not in job:
        raise HTTPException(404, "Result not found")

    path = job["original_preview_path"]

    return stream_temp_file(
        path,
        "image/png",
        f"deepsrm_original_{job_id[:8]}.png"
    )


@app.get("/api/result/{job_id}/preview")
def preview_result(job_id: str):
    job = jobs.get(job_id)

    if not job or "preview_path" not in job:
        raise HTTPException(404, "Result not found")

    path = job["preview_path"]

    return stream_temp_file(
        path,
        "image/png",
        f"deepsrm_preview_{job_id[:8]}.png"
    )

@app.post("/api/upload/process")
async def upload_process(file: UploadFile = File(...)):
    raw = await file.read()
    job_id = str(uuid.uuid4())

    try:
        output = sr_baseline_geotiff(raw, scale=4)
        ai_rgb = ai_rgb_geotiff(raw, scale=4)
        preview = make_preview_from_geotiff(ai_rgb)
        original = source_preview(raw)

        # Store large results on disk, matching the Sentinel processing path.
        geotiff_path = save_temp_result(job_id, "multispectral.tif", output)
        ai_rgb_path = save_temp_result(job_id, "ai_rgb.tif", ai_rgb)
        preview_path = save_temp_result(job_id, "preview.png", preview)
        original_preview_path = save_temp_result(
            job_id,
            "original_preview.png",
            original
        )

        jobs[job_id] = {
            "geotiff_path": geotiff_path,
            "ai_rgb_path": ai_rgb_path,
            "preview_path": preview_path,
            "original_preview_path": original_preview_path,
            "mode": f"{SR_MODEL_NAME}-x4-rgb-plus-multispectral-baseline",
        }

        del output
        del ai_rgb
        del preview
        del original
        gc.collect()

    except Exception as exc:
        gc.collect()
        raise HTTPException(
            400,
            f"Upload must be a readable GeoTIFF for this endpoint: {exc}"
        )

    return {
        "id": job_id,
        "mode": f"{SR_MODEL_NAME}-x4-rgb-plus-multispectral-baseline",
        "download": f"/api/result/{job_id}/download",
        "ai_rgb_download": f"/api/result/{job_id}/ai-rgb-download",
        "preview": f"/api/result/{job_id}/preview",
        "original_preview": f"/api/result/{job_id}/original-preview",
        "neural_model": f"{SR_MODEL_NAME} x4 pretrained RGB visual SR",
    }


