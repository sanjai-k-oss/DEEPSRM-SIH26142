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

EDSR_MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
EDSR_MODEL_PATH = BASE_DIR / "models" / "EDSR_x4.pb"

def ensure_edsr_model():
    """Download the public pretrained EDSR x4 model on first use."""
    if EDSR_MODEL_PATH.exists() and EDSR_MODEL_PATH.stat().st_size > 1_000_000:
        return
    EDSR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(EDSR_MODEL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(EDSR_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def ai_rgb_geotiff(raw_tiff: bytes, scale=4):
    """Run a real pretrained EDSR neural SR model on the RGB view.

    This is an AI visual-SR demonstration, not the final multispectral DEEPSRM model.
    The full multispectral GeoTIFF remains separately available from the Lanczos baseline.
    """
    import gc

    ensure_edsr_model()

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(EDSR_MODEL_PATH))
    sr.setModel("edsr", scale)

    with MemoryFile(raw_tiff) as mem:
        with mem.open() as src:
            rgb = src.read([3, 2, 1]).astype(np.float32)

            np.nan_to_num(rgb, copy=False)

            positives = rgb[rgb > 0]
            p98 = np.percentile(positives, 98) if positives.size else 1.0

            del positives
            gc.collect()

            x = np.clip(
                rgb / max(p98, 1e-6),
                0,
                1
            )

            del rgb
            gc.collect()

            bgr = (
                x[::-1]
                .transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)

            del x
            gc.collect()

            out_bgr = sr.upsample(bgr)

            del bgr
            del sr
            gc.collect()

            out_rgb = (
                out_bgr[:, :, ::-1]
                .transpose(2, 0, 1)
                .astype(np.uint8)
            )

            del out_bgr
            gc.collect()

            h, w = out_rgb.shape[1:]

            profile = src.profile.copy()
            profile.update(
                count=3,
                dtype="uint8",
                height=h,
                width=w,
                transform=src.transform * src.transform.scale(
                    1 / scale,
                    1 / scale
                ),
                compress="deflate"
            )

            result = io.BytesIO()

            with rasterio.open(result, "w", **profile) as dst:
                dst.write(out_rgb)

            output = result.getvalue()

            del out_rgb
            del result
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
    size: int = 512

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
        "mode": "real-copernicus-plus-edsr",
        "sentinel_stac": STAC_URL,
        "process_api": PROCESS_URL,
        "neural_model": "EDSR x4 RGB visual SR; multispectral baseline preserved"
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

    try:
        source = process_sentinel(req)

        # Run EDSR first so the multispectral output is not occupying
        # memory during the neural super-resolution peak.
        ai_rgb = ai_rgb_geotiff(source, scale=4)

        # EDSR no longer needs the source after completion.
        gc.collect()

        # Now create the separate multispectral geospatial baseline.
        multispectral = sr_baseline_geotiff(source, scale=4)

        # Release the raw Sentinel source.
        del source
        gc.collect()

        original = source_preview(ai_rgb)
        enhanced = make_preview_from_geotiff(ai_rgb)

    except Exception as exc:
        gc.collect()
        raise HTTPException(502, f"Sentinel/AI processing failed: {exc}")

    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "geotiff": multispectral,
        "ai_rgb": ai_rgb,
        "preview": enhanced,
        "original_preview": original,
        "mode": "edsr-x4-rgb-plus-multispectral-baseline",
    }

    gc.collect()

    return {
        "id": job_id,
        "mode": "edsr-x4-rgb-plus-multispectral-baseline",
        "input_resolution": "Sentinel-2 L2A 10m/20m source bands",
        "target_resolution": "4× output grid (2.5m for 10m bands; 5m for 20m bands)",
        "neural_model": "EDSR x4 pretrained RGB visual SR",
        "note": "The downloadable multispectral GeoTIFF preserves the Sentinel bands using the geospatial baseline. The AI output is a separate 3-band RGB GeoTIFF. Final DEEPSRM should use a multispectral satellite-trained checkpoint.",
        "download": f"/api/result/{job_id}/download",
        "ai_rgb_download": f"/api/result/{job_id}/ai-rgb-download",
        "preview": f"/api/result/{job_id}/preview",
        "original_preview": f"/api/result/{job_id}/original-preview",
    }

@app.get("/api/result/{job_id}/download")
def download_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Result not found")
    return StreamingResponse(
        io.BytesIO(job["geotiff"]),
        media_type="image/tiff",
        headers={
            "Content-Disposition":
            f'attachment; filename="deepsrm_{job_id[:8]}.tif"'
        },
    )

@app.get("/api/result/{job_id}/ai-rgb-download")
def download_ai_rgb(job_id: str):
    job = jobs.get(job_id)
    if not job or "ai_rgb" not in job:
        raise HTTPException(404, "AI RGB result not found")
    return StreamingResponse(
        io.BytesIO(job["ai_rgb"]), media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="deepsrm_edsr_rgb_{job_id[:8]}.tif"'}
    )

@app.get("/api/result/{job_id}/original-preview")
def original_preview(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Result not found")
    return StreamingResponse(io.BytesIO(job["original_preview"]), media_type="image/png")

@app.get("/api/result/{job_id}/preview")
def preview_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Result not found")
    return StreamingResponse(io.BytesIO(job["preview"]), media_type="image/png")

@app.post("/api/upload/process")
async def upload_process(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        output = sr_baseline_geotiff(raw, scale=4)
        ai_rgb = ai_rgb_geotiff(raw, scale=4)
        preview = make_preview_from_geotiff(ai_rgb)
        original = source_preview(raw)
    except Exception:
        raise HTTPException(
            400,
            "Upload must be a readable GeoTIFF for this endpoint."
        )
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"geotiff": output, "ai_rgb": ai_rgb, "preview": preview, "original_preview": original}
    return {
        "id": job_id,
        "mode": "edsr-x4-rgb-plus-multispectral-baseline",
        "download": f"/api/result/{job_id}/download",
        "ai_rgb_download": f"/api/result/{job_id}/ai-rgb-download",
        "preview": f"/api/result/{job_id}/preview",
        "original_preview": f"/api/result/{job_id}/original-preview",
        "neural_model": "EDSR x4 pretrained RGB visual SR",
    }

