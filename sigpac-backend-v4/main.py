from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import json
import io
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SIGPAC Sentinel API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SIGPAC_CONSULTA_URL   = "https://sigpac-hubcloud.es/servicioconsultassigpac/query"
COPERNICUS_TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COPERNICUS_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# Nueva URL de descarga directa (S3 compatible)
COPERNICUS_S3_URL     = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# STAC API para obtener URLs de assets directamente
STAC_URL              = "https://catalogue.dataspace.copernicus.eu/stac/collections/SENTINEL-2/items"

COPERNICUS_USER = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASS = os.getenv("COPERNICUS_PASS", "")
_token_cache = {"token": None, "expires_at": 0}


def cache_key(prefix: str, **kwargs) -> str:
    key = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(f"{prefix}_{key}".encode()).hexdigest()


async def get_copernicus_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    if not COPERNICUS_USER or not COPERNICUS_PASS:
        raise HTTPException(status_code=500, detail="Credenciales Copernicus no configuradas.")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            COPERNICUS_TOKEN_URL,
            data={"grant_type": "password", "username": COPERNICUS_USER,
                  "password": COPERNICUS_PASS, "client_id": "cdse-public"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        logger.info("Token Copernicus obtenido correctamente")
        return _token_cache["token"]


async def get_band_urls_from_stac(producto_nombre: str, token: str) -> dict:
    """
    Usa la STAC API para obtener las URLs directas de las bandas.
    Devuelve dict: banda -> url
    """
    # El nombre del producto es tipo S2A_MSIL2A_20260427T...
    # Buscamos en STAC por nombre
    stac_search_url = "https://catalogue.dataspace.copernicus.eu/stac/search"
    params = {
        "collections": "SENTINEL-2",
        "filter": f"s2:product_uri = '{producto_nombre}.SAFE'",
        "limit": 1,
    }
    try:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        ) as client:
            resp = await client.post(stac_search_url, json={
                "collections": ["SENTINEL-2"],
                "filter-lang": "cql2-json",
                "filter": {"op": "=", "args": [{"property": "s2:product_uri"}, f"{producto_nombre}.SAFE"]},
                "limit": 1,
            })
            if resp.status_code == 200:
                data = resp.json()
                features = data.get("features", [])
                if features:
                    assets = features[0].get("assets", {})
                    logger.info(f"STAC assets encontrados: {list(assets.keys())[:10]}")
                    return assets
    except Exception as e:
        logger.warning(f"Error STAC: {e}")
    return {}


async def descargar_banda_s3(url: str, token: str) -> Optional[bytes]:
    """Descarga una banda desde URL S3 de Copernicus."""
    try:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=300,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            logger.info(f"Descarga S3 {url[-50:]} → {resp.status_code}")
            if resp.status_code == 200:
                return resp.content
    except Exception as e:
        logger.warning(f"Error descargando S3: {e}")
    return None


async def descargar_bandas_producto(producto_id: str, producto_nombre: str, bandas: list, token: str) -> dict:
    """
    Intenta descargar bandas usando múltiples métodos:
    1. URL directa de descarga del producto completo (ZIP parcial)
    2. STAC API assets
    3. URL alternativa con /download
    """
    arrays = {}

    # Método: URL de descarga directa con el endpoint correcto de Copernicus DS
    base_download = f"https://download.dataspace.copernicus.eu/odata/v1/Products({producto_id})"

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
        follow_redirects=True,
    ) as client:
        for banda in bandas:
            descargado = False

            # Intentos con diferentes estructuras de ruta dentro del SAFE
            rutas = [
                f"{base_download}/Nodes({producto_nombre}.SAFE)/Nodes(GRANULE)/Nodes/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({banda}.jp2)/$value",
                f"{base_download}/Nodes({producto_nombre}.SAFE)/Nodes(GRANULE)/Nodes/Nodes(IMG_DATA)/Nodes(R20m)/Nodes({banda}.jp2)/$value",
                f"{base_download}/Nodes({producto_nombre}.SAFE)/Nodes(GRANULE)/Nodes/Nodes(IMG_DATA)/Nodes({banda}.jp2)/$value",
            ]

            for url in rutas:
                try:
                    resp = await client.get(url)
                    logger.info(f"Intento descarga {banda}: {resp.status_code}")
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        img = Image.open(io.BytesIO(resp.content))
                        arr = np.array(img, dtype=np.float32)
                        if arr.ndim == 3:
                            arr = arr[:, :, 0]
                        arrays[banda] = arr / 10000.0
                        descargado = True
                        logger.info(f"✓ Banda {banda} descargada, shape: {arr.shape}")
                        break
                except Exception as e:
                    logger.warning(f"Error en ruta {banda}: {e}")

            if not descargado:
                logger.warning(f"✗ No se pudo descargar banda {banda}")

    return arrays


def stretch(arr: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip((arr - p2) / (p98 - p2 + 1e-10), 0, 1)
    return (arr * 255).astype(np.uint8)


def igualar_tamanos(arrays: dict) -> dict:
    shapes = [a.shape for a in arrays.values()]
    if len(set(shapes)) <= 1:
        return arrays
    target = min(shapes, key=lambda s: s[0] * s[1])
    return {
        k: np.array(Image.fromarray(v).resize((target[1], target[0]), Image.BILINEAR), dtype=np.float32)
        if v.shape != target else v for k, v in arrays.items()
    }


def calcular_formula(nombre: str, b: dict) -> np.ndarray:
    eps = 1e-10
    if nombre == "NDVI":
        return (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    elif nombre == "NDWI":
        return (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps)
    elif nombre == "EVI":
        return 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6 * b["B04"] - 7.5 * b["B02"] + 1 + eps)
    elif nombre == "NDRE":
        return (b["B08"] - b["B05"]) / (b["B08"] + b["B05"] + eps)
    elif nombre == "SAVI":
        return 1.5 * (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 0.5 + eps)
    raise ValueError(f"Indice desconocido: {nombre}")


INDICES = {
    "NDVI": {"descripcion": "Normalized Difference Vegetation Index", "cmap": "RdYlGn", "vmin": -1, "vmax": 1, "bandas": ["B04", "B08"]},
    "NDWI": {"descripcion": "Normalized Difference Water Index",      "cmap": "Blues",  "vmin": -1, "vmax": 1, "bandas": ["B03", "B08"]},
    "EVI":  {"descripcion": "Enhanced Vegetation Index",              "cmap": "YlGn",   "vmin": -1, "vmax": 1, "bandas": ["B02", "B04", "B08"]},
    "NDRE": {"descripcion": "Normalized Difference Red Edge",         "cmap": "RdYlGn", "vmin": -1, "vmax": 1, "bandas": ["B05", "B08"]},
    "SAVI": {"descripcion": "Soil-Adjusted Vegetation Index",         "cmap": "YlGn",   "vmin": -1, "vmax": 1, "bandas": ["B04", "B08"]},
}


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "5.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "copernicus_configured": bool(COPERNICUS_USER and COPERNICUS_PASS),
    }


@app.get("/sigpac/punto")
async def get_parcela_por_punto(lat: float = Query(...), lon: float = Query(...)):
    ck = cache_key("sigpac_punto", lat=round(lat, 6), lon=round(lon, 6))
    cache_file = CACHE_DIR / f"sigpac_{ck}.geojson"
    if cache_file.exists():
        return JSONResponse(content=json.loads(cache_file.read_text()))

    url = f"{SIGPAC_CONSULTA_URL}/recinfobypoint/4326/{lon}/{lat}.geojson"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        if not data.get("features"):
            raise HTTPException(status_code=404, detail="No se encontró parcela.")
        cache_file.write_text(json.dumps(data))
        return JSONResponse(content=data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error SIGPAC: {str(e)}")


@app.get("/sentinel/buscar")
async def buscar_imagenes(
    bbox: str = Query(...),
    fecha_inicio: str = Query(...),
    fecha_fin: str = Query(...),
    max_nubosidad: float = Query(30.0),
):
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox invalido")

    footprint = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and ContentDate/Start gt {fecha_inicio}T00:00:00.000Z "
            f"and ContentDate/Start lt {fecha_fin}T23:59:59.000Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_nubosidad}) "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": "10",
        "$expand": "Attributes",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(COPERNICUS_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        productos = []
        for item in data.get("value", []):
            cloud = next((a["Value"] for a in item.get("Attributes", []) if a["Name"] == "cloudCover"), None)
            productos.append({
                "id": item["Id"],
                "nombre": item["Name"],
                "fecha": item["ContentDate"]["Start"][:10],
                "nubosidad": round(cloud, 1) if cloud is not None else None,
                "size_mb": round(item.get("ContentLength", 0) / 1e6, 1),
            })
        return {"total": len(productos), "productos": productos}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error Copernicus: {e}")


@app.get("/imagen/rgb")
async def imagen_rgb(
    producto_id: str = Query(...),
    producto_nombre: str = Query(..., description="Nombre del producto ej: S2A_MSIL2A_20260427T..."),
    bbox: Optional[str] = Query(None),
):
    ck = cache_key("rgb5", pid=producto_id, bbox=bbox or "")
    cache_png = CACHE_DIR / f"{ck}_rgb.png"

    if cache_png.exists():
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_rgb(cache_png)

    arrays = await descargar_bandas_producto(producto_id, producto_nombre, ["B04", "B03", "B02"], token)

    if len(arrays) < 3:
        logger.warning(f"Solo se descargaron {len(arrays)} bandas de 3, usando demo")
        return _demo_rgb(cache_png)

    arrays = igualar_tamanos(arrays)
    rgb = np.stack([stretch(arrays["B04"]), stretch(arrays["B03"]), stretch(arrays["B02"])], axis=2)
    img = Image.fromarray(rgb, mode='RGB')

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/indices/lista")
async def lista_indices():
    return {k: {"descripcion": v["descripcion"], "bandas": v["bandas"]} for k, v in INDICES.items()}


@app.get("/indice/calcular")
async def calcular_indice(
    producto_id: str = Query(...),
    producto_nombre: str = Query(...),
    indice: str = Query(...),
    bbox: Optional[str] = Query(None),
    formato: str = Query("png"),
):
    indice = indice.upper()
    if indice not in INDICES:
        raise HTTPException(status_code=400, detail=f"Indice desconocido: {list(INDICES.keys())}")

    cfg = INDICES[indice]
    ck = cache_key("indice5", pid=producto_id, idx=indice, bbox=bbox or "")
    cache_png = CACHE_DIR / f"{ck}.png"
    cache_stats = CACHE_DIR / f"{ck}_stats.json"

    if cache_png.exists() and formato == "png":
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")
    if cache_stats.exists() and formato == "stats":
        return JSONResponse(content=json.loads(cache_stats.read_text()))

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_indice(indice, cfg, cache_png, cache_stats, formato)

    arrays = await descargar_bandas_producto(producto_id, producto_nombre, cfg["bandas"], token)

    if len(arrays) < len(cfg["bandas"]):
        return _demo_indice(indice, cfg, cache_png, cache_stats, formato)

    arrays = igualar_tamanos(arrays)
    resultado = np.clip(calcular_formula(indice, arrays), cfg["vmin"], cfg["vmax"])
    return _render_indice(resultado, indice, cfg, cache_png, cache_stats, formato, demo=False)


def _demo_rgb(cache_png: Path):
    np.random.seed(123)
    size = (256, 256)
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    base = 0.5 + 0.3 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2)
    r = np.clip(base * 80 + np.random.normal(0, 5, size), 50, 130).astype(np.uint8)
    g = np.clip(base * 120 + np.random.normal(0, 5, size), 80, 180).astype(np.uint8)
    b = np.clip(base * 50 + np.random.normal(0, 5, size), 30, 90).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


def _demo_indice(indice, cfg, cache_png, cache_stats, formato):
    np.random.seed(42)
    x, y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    base = 0.3 + 0.4 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.15)
    resultado = np.clip(base + np.random.normal(0, 0.05, (256, 256)), cfg["vmin"], cfg["vmax"]).astype(np.float32)
    return _render_indice(resultado, indice, cfg, cache_png, cache_stats, formato, demo=True)


def _render_indice(resultado, indice, cfg, cache_png, cache_stats, formato, demo=False):
    stats = {
        "indice": indice,
        "min": float(np.nanmin(resultado)),
        "max": float(np.nanmax(resultado)),
        "mean": float(np.nanmean(resultado)),
        "std": float(np.nanstd(resultado)),
    }
    if demo:
        stats["modo"] = "DEMO"
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    titulo = f"{indice} - {cfg['descripcion']}" + (" (DEMO)" if demo else "")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('#0a0f0d')
    ax.set_facecolor('#0a1a0d')
    im = ax.imshow(resultado, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#6b8f72')
    cbar.outline.set_edgecolor('#2a3d2e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#6b8f72', fontsize=9)
    ax.set_title(titulo, color='#e2ffe8', fontsize=12, fontweight='bold', pad=12)
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0f0d')
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


@app.get("/cache/info")
async def cache_info():
    files = list(CACHE_DIR.glob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    return {"archivos": len(files), "total_mb": round(total_mb, 2)}


@app.delete("/cache/limpiar")
async def limpiar_cache(dias: int = Query(7)):
    cutoff = time.time() - dias * 86400
    eliminados = sum(1 for f in CACHE_DIR.glob("*") if f.is_file() and f.stat().st_mtime < cutoff and not f.unlink())
    return {"eliminados": eliminados}
