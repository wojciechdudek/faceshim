"""faceshim - a DeepStack-compatible face recognition API on InsightFace.

Double Take (and anything else that speaks the DeepStack face API) can use it as
a drop-in "deepstack" detector:

    POST /v1/vision/face/register   image, userid          -> {"success": true}
    POST /v1/vision/face/recognize  image[, min_confidence] -> {"success": true, "predictions": [...]}
    POST /v1/vision/face/list                               -> {"success": true, "faces": [...]}
    POST /v1/vision/face/delete     userid                  -> {"success": true}
    GET  /                                                  -> health + stats

Embeddings live in one JSON file under DATA_DIR; the model is loaded once at
startup, so the first request after a restart is as fast as any other.
"""

import asyncio
import json
import logging
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")  # skimage API churn, harmless
log = logging.getLogger("faceshim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

MODEL_NAME = os.getenv("MODEL_NAME", "buffalo_l")
DET_SIZE = int(os.getenv("DET_SIZE", "640"))
# Cosine similarity -> confidence is linear between these two points:
# SIM_LOW maps to 0 %, SIM_HIGH to 100 %. Tune them on your own camera - see README.
SIM_LOW = float(os.getenv("SIM_LOW", "0.30"))
SIM_HIGH = float(os.getenv("SIM_HIGH", "0.70"))
# Below this calibrated confidence a face is reported as "unknown".
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.20"))
MIN_FACE_PX = int(os.getenv("MIN_FACE_PX", "40"))
# Tightly cropped portraits (a head filling the frame, hair cut off) make SCRFD miss or doubt the
# face. Below this det_score the image is padded with a neutral border and detected again.
PAD_RETRY_BELOW = float(os.getenv("PAD_RETRY_BELOW", "0.80"))
PAD_FRACTION = float(os.getenv("PAD_FRACTION", "0.5"))
MAX_FACES = int(os.getenv("MAX_FACES", "5"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
PROVIDERS = [p.strip() for p in os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]


def sim_to_conf(sim: float) -> float:
    return float(min(1.0, max(0.0, (sim - SIM_LOW) / (SIM_HIGH - SIM_LOW))))


class Store:
    """Per-user list of L2-normalised embeddings, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.users: dict[str, list[np.ndarray]] = {}
        if path.exists():
            data = json.loads(path.read_text())
            self.users = {
                user: [np.asarray(e, dtype=np.float32) for e in embs]
                for user, embs in data.get("users", {}).items()
            }
            log.info("loaded %d users / %d embeddings from %s", len(self.users), self.count(), path)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "version": 1,
            "model": MODEL_NAME,
            "users": {u: [e.tolist() for e in embs] for u, embs in self.users.items()},
        }))
        os.replace(tmp, self.path)

    def count(self) -> int:
        return sum(len(v) for v in self.users.values())

    def names(self) -> list[str]:
        return sorted(self.users)

    def add(self, user: str, emb: np.ndarray) -> int:
        with self.lock:
            embs = self.users.setdefault(user, [])
            embs.append(emb.astype(np.float32))
            try:
                self._save()
            except OSError:
                embs.pop()  # keep memory and disk consistent: a failed write enrolls nobody
                if not embs:
                    del self.users[user]
                raise
            return len(embs)

    def delete(self, user: str) -> bool:
        with self.lock:
            if user not in self.users:
                return False
            del self.users[user]
            self._save()
            return True

    def best_match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        """Highest cosine similarity across every stored embedding, per user."""
        best_user, best_sim = None, -1.0
        with self.lock:
            for user, embs in self.users.items():
                sim = float(np.max(np.stack(embs) @ emb))
                if sim > best_sim:
                    best_user, best_sim = user, sim
        return best_user, best_sim


class Engine:
    def __init__(self):
        from insightface.app import FaceAnalysis  # slow import - keep it out of module load

        self.app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS, allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1 if PROVIDERS == ["CPUExecutionProvider"] else 0, det_size=(DET_SIZE, DET_SIZE))
        self.lock = threading.Lock()

    def _detect(self, img: np.ndarray):
        with self.lock:
            faces = self.app.get(img)
        faces = [f for f in faces if (f.bbox[2] - f.bbox[0]) >= MIN_FACE_PX and (f.bbox[3] - f.bbox[1]) >= MIN_FACE_PX]
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    def faces(self, img: np.ndarray):
        faces = self._detect(img)
        if PAD_FRACTION > 0 and (not faces or faces[0].det_score < PAD_RETRY_BELOW):
            pad = int(PAD_FRACTION * max(img.shape[:2]))
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            retry = self._detect(padded)
            if retry and (not faces or retry[0].det_score > faces[0].det_score):
                for f in retry:  # embeddings were computed on the padded image; only coordinates move back
                    f.bbox = f.bbox - pad
                    f.kps = f.kps - pad
                log.info("padded retry: det %.2f -> %.2f", faces[0].det_score if faces else 0.0, retry[0].det_score)
                faces = retry
        return faces[:MAX_FACES]


engine: Engine
store: Store


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, store
    t0 = time.time()
    engine = Engine()
    store = Store(DATA_DIR / "faces.json")
    check_writable(DATA_DIR)
    log.info("model %s ready in %.1fs, providers=%s", MODEL_NAME, time.time() - t0, PROVIDERS)
    yield


def check_writable(path: Path) -> None:
    """Fail fast: a bind mount owned by root would otherwise break every register with a 500."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        log.error("%s is not writable by uid %d (%s). On the host: chown -R %d:%d <mounted dir>, "
                  "or use a named volume.", path, os.getuid(), e, os.getuid(), os.getgid())
        raise SystemExit(1)


app = FastAPI(title="faceshim", lifespan=lifespan)


def decode(data: bytes) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def bbox(face) -> dict:
    x1, y1, x2, y2 = (int(round(float(v))) for v in face.bbox)
    return {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}


@app.get("/")
async def health():
    return {"success": True, "model": MODEL_NAME, "providers": PROVIDERS,
            "users": store.names(), "embeddings": store.count()}


@app.post("/v1/vision/face/register")
async def register(image: UploadFile = File(...), userid: str = Form(...)):
    userid = userid.strip()
    img = decode(await image.read())
    if img is None:
        return {"success": False, "error": "invalid image"}
    faces = await asyncio.to_thread(engine.faces, img)
    if not faces:
        return {"success": False, "error": "no face found"}
    face = faces[0]  # largest face in the picture is the one being enrolled
    try:
        n = store.add(userid, face.normed_embedding)
    except OSError as e:
        log.error("register %s failed: cannot write %s: %s", userid, DATA_DIR, e)
        return {"success": False, "error": f"cannot write {DATA_DIR}: {e}"}
    log.info("register %s: face %s det=%.2f (%d embeddings)", userid, bbox(face), face.det_score, n)
    return {"success": True, "message": "face added", "userid": userid, "embeddings": n, **bbox(face)}


@app.post("/v1/vision/face/recognize")
async def recognize(image: UploadFile = File(...), min_confidence: Optional[float] = Form(None)):
    t0 = time.time()
    img = decode(await image.read())
    if img is None:
        return {"success": False, "error": "invalid image"}
    floor = MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
    faces = await asyncio.to_thread(engine.faces, img)
    predictions = []
    for face in faces:
        user, sim = store.best_match(face.normed_embedding)
        conf = sim_to_conf(sim) if user else 0.0
        known = user is not None and conf >= floor
        predictions.append({
            "userid": user if known else "unknown",
            "confidence": round(conf if known else 0.0, 4),
            # Extra fields (ignored by Double Take) for calibration and debugging:
            "similarity": round(sim, 4),
            "candidate": user,
            "det_score": round(float(face.det_score), 4),
            **bbox(face),
        })
    ms = int((time.time() - t0) * 1000)
    log.info("recognize: %d face(s) in %dms %s", len(predictions), ms,
             [(p["userid"], p["confidence"], p["similarity"]) for p in predictions])
    return {"success": True, "predictions": predictions, "duration": ms}


@app.api_route("/v1/vision/face/list", methods=["GET", "POST"])
async def list_faces():
    return {"success": True, "faces": store.names()}


@app.post("/v1/vision/face/delete")
async def delete(userid: str = Form(...)):
    ok = store.delete(userid.strip())
    return {"success": ok, **({} if ok else {"error": "unknown userid"})}
