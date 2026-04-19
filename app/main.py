from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .rag_engine import RagEngine
from .schemas import AskRequest, AskResponse, HealthResponse


app = FastAPI(title="Wiki-CN QA Web", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

_sessions_store: dict[str, list[dict[str, str]]] = {}
_session_lock = threading.Lock()

engine = None
_engine_thread: threading.Thread | None = None
_engine_lock = threading.Lock()
_RETRY_BASE_SECONDS = 10.0
_RETRY_MAX_SECONDS = 180.0
_STAGE_ORDER = ["loading_chunks", "loading_bm25", "loading_faiss", "initializing_client"]
_DEFAULT_STAGE_EST_SEC = {
    "loading_chunks": 180.0,
    "loading_bm25": 240.0,
    "loading_faiss": 300.0,
    "initializing_client": 6.0,
}
_PRED_INTERVAL_FACTORS = {
    "low": (0.6, 1.8),
    "medium": (0.75, 1.4),
    "high": (0.85, 1.2),
}
_engine_state = {
    "phase": "idle",  # idle | loading | retry_wait | ready
    "stage": "idle",  # loading_chunks | loading_bm25 | loading_faiss | initializing_client | ready
    "started_at": 0.0,
    "ready_at": 0.0,
    "error": "",
    "retry_count": 0,
    "next_retry_at": 0.0,
    "stage_started_at": 0.0,
    "stage_durations": {},
    "stage_estimates_sec": dict(_DEFAULT_STAGE_EST_SEC),
}


def _engine_state_snapshot() -> dict:
    with _engine_lock:
        return dict(_engine_state)


def _set_engine_stage(stage: str) -> None:
    with _engine_lock:
        now = time.time()
        prev_stage = str(_engine_state.get("stage", "idle"))
        prev_started_at = float(_engine_state.get("stage_started_at", 0.0) or 0.0)
        if prev_started_at > 0.0 and prev_stage:
            elapsed = max(0.0, now - prev_started_at)
            durations = dict(_engine_state.get("stage_durations", {}))
            durations[prev_stage] = round(float(durations.get(prev_stage, 0.0)) + elapsed, 2)
            _engine_state["stage_durations"] = durations

            if prev_stage in _STAGE_ORDER:
                est = dict(_engine_state.get("stage_estimates_sec", {}))
                old = float(est.get(prev_stage, _DEFAULT_STAGE_EST_SEC.get(prev_stage, elapsed)))
                # 用 EMA 平滑预测，避免单次抖动导致 ETA 跳变。
                est[prev_stage] = round(old * 0.7 + elapsed * 0.3, 2)
                _engine_state["stage_estimates_sec"] = est
        _engine_state["stage"] = stage
        _engine_state["stage_started_at"] = now


def _load_engine_worker() -> None:
    global engine
    attempt = 0
    while True:
        attempt += 1
        with _engine_lock:
            now = time.time()
            if _engine_state["started_at"] <= 0.0:
                _engine_state["started_at"] = now
                _engine_state["stage_durations"] = {}
                _engine_state["stage_estimates_sec"] = dict(_DEFAULT_STAGE_EST_SEC)
            _engine_state["phase"] = "loading"
            _engine_state["stage"] = "booting"
            _engine_state["stage_started_at"] = now
            _engine_state["ready_at"] = 0.0
            _engine_state["error"] = ""
            _engine_state["retry_count"] = attempt - 1
            _engine_state["next_retry_at"] = 0.0

        try:
            built = RagEngine(on_stage=_set_engine_stage)
            with _engine_lock:
                now = time.time()
                engine = built
                _engine_state["phase"] = "ready"
                _engine_state["stage"] = "ready"
                _engine_state["stage_started_at"] = now
                _engine_state["ready_at"] = now
                _engine_state["error"] = ""
                _engine_state["retry_count"] = attempt - 1
                _engine_state["next_retry_at"] = 0.0
            return
        except Exception as e:
            wait_s = min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1)))
            next_retry_at = time.time() + wait_s
            with _engine_lock:
                engine = None
                _engine_state["phase"] = "retry_wait"
                _engine_state["stage"] = "retry_wait"
                _engine_state["error"] = str(e)
                _engine_state["retry_count"] = attempt
                _engine_state["next_retry_at"] = next_retry_at
            time.sleep(wait_s)


def _ensure_engine_loading() -> None:
    global _engine_thread
    with _engine_lock:
        phase = _engine_state.get("phase", "idle")
        alive = _engine_thread is not None and _engine_thread.is_alive()
        if phase in {"loading", "retry_wait", "ready"} or alive:
            return
        _engine_thread = threading.Thread(target=_load_engine_worker, name="rag-engine-loader", daemon=True)
        _engine_thread.start()


@app.on_event("startup")
def startup_event():
    _ensure_engine_loading()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    return RedirectResponse(url="/static/favicon.svg")


@app.post("/api/ask", response_model=AskResponse)
def ask_api(payload: AskRequest):
    _ensure_engine_loading()

    q = payload.question.strip()
    if not q:
        return AskResponse(answer="请输入问题。", references=[])

    if engine is None:
        state = _engine_state_snapshot()
        if state.get("phase") == "retry_wait":
            retry_in = max(0.0, float(state.get("next_retry_at", 0.0)) - time.time())
            return AskResponse(answer=f"知识库加载失败，系统将在 {retry_in:.1f}s 后自动重试。", references=[])
        return AskResponse(answer="知识库正在后台加载中，请稍后再试。", references=[])

    session_id = payload.session_id
    with _session_lock:
        if session_id not in _sessions_store:
            _sessions_store[session_id] = []
        history = _sessions_store[session_id][-payload.history_len * 2:] if payload.history_len > 0 else []

    answer, refs = engine.ask(q, history=history, allow_web=payload.allow_web)

    with _session_lock:
        if session_id in _sessions_store:
            _sessions_store[session_id].append({"role": "user", "content": q})
            _sessions_store[session_id].append({"role": "assistant", "content": answer})
            # keep an upper bound on memory to prevent leak
            if len(_sessions_store[session_id]) > 100:
                _sessions_store[session_id] = _sessions_store[session_id][-50:]

    return AskResponse(answer=answer, references=refs)


@app.get("/api/health", response_model=HealthResponse)
def health_api():
    _ensure_engine_loading()

    now = time.time()
    state = _engine_state_snapshot()
    phase = str(state.get("phase", "idle"))
    stage = str(state.get("stage", "idle"))
    started_at = float(state.get("started_at", 0.0) or 0.0)
    ready_at = float(state.get("ready_at", 0.0) or 0.0)
    next_retry_at = float(state.get("next_retry_at", 0.0) or 0.0)
    stage_started_at = float(state.get("stage_started_at", 0.0) or 0.0)
    stage_durations = dict(state.get("stage_durations", {}) or {})
    stage_estimates = dict(state.get("stage_estimates_sec", {}) or {})
    retry_count = int(state.get("retry_count", 0) or 0)
    elapsed = round(max(0.0, now - started_at), 2) if started_at > 0 else 0.0
    retry_in = round(max(0.0, next_retry_at - now), 2) if next_retry_at > 0 else 0.0
    stage_elapsed = round(max(0.0, now - stage_started_at), 2) if stage_started_at > 0 else 0.0

    current_est = float(stage_estimates.get(stage, _DEFAULT_STAGE_EST_SEC.get(stage, 0.0)))
    stage_remaining = round(max(0.0, current_est - stage_elapsed), 2) if stage in _STAGE_ORDER else 0.0

    total_est = 0.0
    total_remaining = 0.0
    completed_stage_count = 0
    for s in _STAGE_ORDER:
        est_s = float(stage_estimates.get(s, _DEFAULT_STAGE_EST_SEC.get(s, 0.0)))
        total_est += est_s

        if s in stage_durations:
            completed_stage_count += 1
            continue
        if s == stage and phase == "loading":
            total_remaining += max(0.0, est_s - stage_elapsed)
        elif phase == "ready":
            continue
        else:
            total_remaining += est_s

    if completed_stage_count >= 3:
        pred_conf = "high"
    elif completed_stage_count >= 1:
        pred_conf = "medium"
    else:
        pred_conf = "low"
    low_factor, high_factor = _PRED_INTERVAL_FACTORS[pred_conf]

    stage_remaining_p50 = round(stage_remaining, 2)
    stage_remaining_p90 = round(stage_remaining * high_factor, 2)
    total_remaining_p50 = round(max(0.0, total_remaining), 2)
    total_remaining_p90 = round(max(0.0, total_remaining) * high_factor, 2)

    if engine is not None:
        payload = engine.health_status()
    else:
        if phase == "loading":
            status = "loading"
        elif phase == "retry_wait":
            status = "degraded"
        else:
            status = "starting"
        payload = {
            "status": status,
            "client_ready": False,
            "vector_enabled": False,
            "llm_enabled": False,
            "dense_backend": "unknown",
            "force_bm25_only": False,
            "dense_ready": False,
            "dense_disabled_reason": "",
            "llm_primary_provider": "unknown",
            "llm_fallback_enabled": False,
            "llm_provider_last": "extractive",
            "web_fallback_enabled": False,
            "vector_cooldown_left_sec": 0.0,
            "llm_cooldown_left_sec": 0.0,
            "total_chunks": 0,
            "index_ntotal": 0,
            "last_bm25_hits": 0,
            "last_vec_hits": 0,
            "last_vec_used": False,
            "last_rerank_used": False,
            "module_statuses": {
                "chunks": "loading" if phase == "loading" else "pending",
                "bm25": "loading" if phase == "loading" else "pending",
                "dense": "pending",
                "rerank": "pending",
                "llm_primary": "pending",
                "llm_fallback": "pending",
                "web_fallback": "disabled",
            },
        }

    payload.update(
        {
            "engine_phase": phase,
            "engine_stage": stage,
            "engine_loading": phase == "loading",
            "engine_ready": phase == "ready" and engine is not None,
            "engine_started_at": started_at,
            "engine_ready_at": ready_at,
            "engine_load_elapsed_sec": elapsed,
            "engine_stage_started_at": stage_started_at,
            "engine_stage_elapsed_sec": stage_elapsed,
            "engine_stage_durations": stage_durations,
            "engine_stage_estimates_sec": {k: round(float(v), 2) for k, v in stage_estimates.items()},
            "engine_stage_remaining_sec": stage_remaining,
            "engine_total_estimated_sec": round(total_est, 2),
            "engine_total_remaining_sec": round(max(0.0, total_remaining), 2),
            "engine_prediction_confidence": pred_conf,
            "engine_stage_remaining_sec_p50": stage_remaining_p50,
            "engine_stage_remaining_sec_p90": stage_remaining_p90,
            "engine_total_remaining_sec_p50": total_remaining_p50,
            "engine_total_remaining_sec_p90": total_remaining_p90,
            "engine_error": str(state.get("error", "")),
            "engine_retry_count": retry_count,
            "engine_next_retry_at": next_retry_at,
            "engine_retry_in_sec": retry_in,
        }
    )

    return HealthResponse(**payload)
