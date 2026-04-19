from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    allow_web: bool = False
    session_id: str = "default"
    history_len: int = 5


class ReferenceItem(BaseModel):
    doc_id: str
    title: str
    url: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    references: list[ReferenceItem]


class HealthResponse(BaseModel):
    status: str
    client_ready: bool
    vector_enabled: bool
    llm_enabled: bool
    dense_backend: str = "faiss"
    force_bm25_only: bool = False
    dense_ready: bool = False
    dense_disabled_reason: str = ""
    llm_primary_provider: str = "ollama"
    llm_fallback_enabled: bool = False
    llm_provider_last: str = "extractive"
    web_fallback_enabled: bool = False
    vector_cooldown_left_sec: float
    llm_cooldown_left_sec: float
    total_chunks: int
    index_ntotal: int
    last_bm25_hits: int = 0
    last_vec_hits: int = 0
    last_vec_used: bool = False
    last_rerank_used: bool = False
    module_statuses: dict[str, str] = Field(default_factory=dict)
    engine_phase: str = "idle"
    engine_stage: str = "idle"
    engine_loading: bool = False
    engine_ready: bool = False
    engine_started_at: float = 0.0
    engine_ready_at: float = 0.0
    engine_load_elapsed_sec: float = 0.0
    engine_stage_started_at: float = 0.0
    engine_stage_elapsed_sec: float = 0.0
    engine_stage_durations: dict[str, float] = Field(default_factory=dict)
    engine_stage_estimates_sec: dict[str, float] = Field(default_factory=dict)
    engine_stage_remaining_sec: float = 0.0
    engine_total_estimated_sec: float = 0.0
    engine_total_remaining_sec: float = 0.0
    engine_prediction_confidence: str = "low"
    engine_stage_remaining_sec_p50: float = 0.0
    engine_stage_remaining_sec_p90: float = 0.0
    engine_total_remaining_sec_p50: float = 0.0
    engine_total_remaining_sec_p90: float = 0.0
    engine_error: str = ""
    engine_retry_count: int = 0
    engine_next_retry_at: float = 0.0
    engine_retry_in_sec: float = 0.0
