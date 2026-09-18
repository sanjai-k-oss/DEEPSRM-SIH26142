# DEEPSRM SIH26142 — Real Sentinel-2 + AI SR v3

This version adds a **real pretrained neural super-resolution step** to the live Sentinel-2 prototype.

## What is real
1. Search Sentinel-2 L2A scenes through the Copernicus Data Space Ecosystem Catalog/STAC.
2. Retrieve B02/B03/B04/B05 + SCL through the Copernicus Sentinel Hub Process API.
3. Mask cloud/cloud-shadow classes.
4. Create a georeferenced multispectral GeoTIFF.
5. Run a pretrained **EDSR ×4** neural super-resolution model on the RGB visualization.
6. Export a separate AI RGB GeoTIFF with the correct geospatial transform.
7. Keep a separate 5-band multispectral GeoTIFF produced with a geospatial Lanczos baseline.

Copernicus documents Sentinel-2 L2A access through the Process API and the `sentinel-2-l2a` data type; the Process API supports custom band selection and GeoTIFF output. See the official documentation for details.

## Important scientific limitation
EDSR is a general image super-resolution model, not a satellite-trained multispectral DEEPSRM model. Therefore:

- The **AI preview / AI RGB GeoTIFF is a visual demonstration** of neural SR.
- The **multispectral GeoTIFF is not claimed to be AI-super-resolved**.
- Do not claim PSNR/SSIM/SAM results unless you evaluate against an appropriate high-resolution reference dataset.
- Do not describe generated 2.5 m pixels as guaranteed physical 2.5 m satellite observations. They are an inferred higher-resolution representation.

The final SIH research version should replace EDSR with a model trained/fine-tuned on paired satellite imagery, with multispectral/spectral-consistency losses and validation against higher-resolution reference imagery.

## Run

### 1. Install

```bash
cd backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Copernicus

Copy `.env.example` to `.env` and fill in your Copernicus Data Space OAuth credentials.

### 3. Start API

```bash
uvicorn main:app --reload --port 8000
```

### 4. Start frontend

In another terminal:

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500`.

### First AI run

The backend downloads the public EDSR ×4 model into `backend/models/EDSR_x4.pb` if it is not already present. The download requires internet access.

## Output

- **Enhanced Preview:** EDSR ×4 neural RGB output.
- **AI RGB GeoTIFF:** 3-band georeferenced AI output.
- **Multispectral GeoTIFF:** B02/B03/B04/B05/SCL-derived georeferenced output using the baseline resampling path.

## Why this is not yet the final DEEPSRM model

The SIH deck mentions SwinIR/ESRGAN and spectral preservation. This v3 deliberately does not pretend that a natural-image EDSR checkpoint is a satellite-trained DEEPSRM model. The next research step is a satellite-specific checkpoint trained on suitable paired data, then plugging that checkpoint into the same `/api/sentinel/process` pipeline.
