"""
IntelliPipe FastAPI Backend
=============================
Production-grade REST + WebSocket API with:
- JWT authentication
- Rate limiting
- Request tracing
- WebSocket real-time DQ score streaming
- Comprehensive error handling
- OpenAPI documentation
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.core.config import get_settings
from src.core.logging import (
    configure_logging,
    get_logger,
    set_request_id,
    set_tenant_id,
)
from src.core.telemetry import (
    API_REQUEST_DURATION,
    API_REQUESTS_IN_FLIGHT,
    setup_telemetry,
)

logger = get_logger(__name__, component="api")
settings = get_settings()

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------


class TokenRequest(BaseModel):
    username: str
    password: str
    tenant_id: str = "default"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class IncidentResponse(BaseModel):
    id: str
    title: str
    severity: str
    anomaly_type: str
    status: str
    table_name: str
    anomaly_score: Optional[float]
    root_cause_analysis: Optional[str]
    github_pr_url: Optional[str]
    jira_ticket_key: Optional[str]
    created_at: str
    resolved_at: Optional[str]


class DQScoreResponse(BaseModel):
    table: str
    tenant_id: str
    overall: float
    completeness: float
    validity: float
    uniqueness: float
    freshness: float
    consistency: float
    row_count: int
    computed_at: str


class IncidentListResponse(BaseModel):
    incidents: List[IncidentResponse]
    total: int
    page: int
    page_size: int


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    source_type: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    chunks_retrieved: int


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    checks: Dict[str, str]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle hooks."""
    configure_logging(
        log_level=settings.observability.log_level,
        json_output=settings.is_production,
        service_name=settings.observability.service_name,
        service_version=settings.observability.service_version,
        environment=settings.environment,
    )

    if settings.is_production:
        setup_telemetry(
            service_name=settings.observability.service_name,
            service_version=settings.observability.service_version,
            environment=settings.environment,
            otlp_endpoint=settings.observability.exporter_otlp_endpoint,
            prometheus_port=settings.observability.prometheus_port,
        )

    # Initialise Redis connection pool
    app.state.redis = await aioredis.from_url(
        settings.redis.url,
        max_connections=settings.redis.max_connections,
        decode_responses=True,
    )

    logger.info(
        "IntelliPipe API started",
        version=settings.app_version,
        env=settings.environment,
    )
    yield

    await app.state.redis.close()
    logger.info("IntelliPipe API shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IntelliPipe API",
    description="Autonomous Data Quality & Anomaly Intelligence Platform",
    version=settings.app_version,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# Middleware stack
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next: Any) -> Response:
    """Inject request ID and tenant ID into every request context."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    tenant_id = request.headers.get("X-Tenant-ID", "default")

    set_request_id(request_id)
    set_tenant_id(tenant_id)

    start = time.perf_counter()
    endpoint = request.url.path

    API_REQUESTS_IN_FLIGHT.labels(endpoint=endpoint).inc()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start
        API_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).observe(duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
        return response
    finally:
        API_REQUESTS_IN_FLIGHT.labels(endpoint=endpoint).dec()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

security = HTTPBearer()

ALGORITHM = "HS256"
SECRET_KEY = settings.api.secret_key.get_secret_value()

# Dummy user store — replace with real user repo in production
_USERS: Dict[str, Dict[str, Any]] = {
    "admin": {"password": "intellipipe-admin", "tenant_id": "default", "role": "admin"},
    "viewer": {
        "password": "intellipipe-view",
        "tenant_id": "default",
        "role": "viewer",
    },
}


def create_access_token(data: Dict[str, Any], expires_minutes: int) -> str:
    payload = {
        **data,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {
            "username": username,
            "tenant_id": payload.get("tenant_id", "default"),
            "role": payload.get("role", "viewer"),
        }
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")


def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(req: TokenRequest) -> TokenResponse:
    user = _USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    expires = settings.api.access_token_expire_minutes
    token = create_access_token(
        {"sub": req.username, "tenant_id": req.tenant_id, "role": user["role"]},
        expires,
    )
    return TokenResponse(access_token=token, expires_in=expires * 60)


# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    checks: Dict[str, str] = {}

    # Redis check
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "healthy"
    except Exception:
        checks["redis"] = "unhealthy"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
    )


@app.get("/ready", tags=["System"])
async def readiness_probe() -> Dict[str, str]:
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# DQ Score endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/dq/scores", response_model=List[DQScoreResponse], tags=["Data Quality"]
)
@limiter.limit("200/minute")
async def get_all_dq_scores(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
) -> List[DQScoreResponse]:
    """Get latest DQ scores for all tables in the tenant."""
    redis_client = request.app.state.redis
    tenant_id = user["tenant_id"]

    pattern = f"{settings.redis.dq_scores_key}:{tenant_id}:*"
    keys = await redis_client.keys(pattern)

    scores = []
    for key in keys:
        raw = await redis_client.get(key)
        if raw:
            data = json.loads(raw)
            scores.append(
                DQScoreResponse(
                    table=data.get("table", ""),
                    tenant_id=data.get("tenant_id", tenant_id),
                    overall=data.get("overall", 0.0),
                    completeness=data.get("completeness", 0.0),
                    validity=data.get("validity", 0.0),
                    uniqueness=data.get("uniqueness", 0.0),
                    freshness=data.get("freshness", 0.0),
                    consistency=data.get("consistency", 0.0),
                    row_count=int(data.get("row_count", 0)),
                    computed_at=data.get("computed_at", ""),
                )
            )

    return scores


@app.get(
    "/api/v1/dq/scores/{table_name}",
    response_model=DQScoreResponse,
    tags=["Data Quality"],
)
async def get_dq_score(
    table_name: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
) -> DQScoreResponse:
    """Get latest DQ score for a specific table."""
    redis_client = request.app.state.redis
    tenant_id = user["tenant_id"]
    key = f"{settings.redis.dq_scores_key}:{tenant_id}:{table_name}"
    raw = await redis_client.get(key)

    if not raw:
        raise HTTPException(
            status_code=404, detail=f"No DQ score found for table '{table_name}'"
        )

    data = json.loads(raw)
    return DQScoreResponse(
        **{k: data[k] for k in DQScoreResponse.model_fields if k in data}
    )


# ---------------------------------------------------------------------------
# Incident endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/incidents", response_model=IncidentListResponse, tags=["Incidents"])
@limiter.limit("100/minute")
async def list_incidents(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    severity: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
) -> IncidentListResponse:
    """List incidents for the tenant with pagination."""
    # In production: query IncidentRepository
    # Returning mock data structure for API contract validation
    return IncidentListResponse(
        incidents=[],
        total=0,
        page=page,
        page_size=page_size,
    )


@app.get(
    "/api/v1/incidents/{incident_id}",
    response_model=IncidentResponse,
    tags=["Incidents"],
)
async def get_incident(
    incident_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> IncidentResponse:
    """Get full incident details by ID."""
    raise HTTPException(status_code=404, detail="Incident not found")


@app.post("/api/v1/incidents/{incident_id}/resolve", tags=["Incidents"])
async def resolve_incident(
    incident_id: str,
    resolution_notes: Optional[str] = None,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, str]:
    """Mark an incident as resolved."""
    logger.info(
        "Incident resolved", incident_id=incident_id, resolved_by=user["username"]
    )
    return {"status": "resolved", "incident_id": incident_id}


# ---------------------------------------------------------------------------
# RAG / Docs query endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
@limiter.limit("30/minute")
async def query_docs(
    request: Request,
    body: RAGQueryRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> RAGQueryResponse:
    """Natural language query over dbt docs, lineage, and data contracts."""
    logger.info(
        "RAG query received", question=body.question[:80], tenant=user["tenant_id"]
    )
    # In production: call RAGEngine.query(...)
    return RAGQueryResponse(
        answer="RAG engine not yet connected to this endpoint in dev mode.",
        sources=[],
        confidence=0.0,
        chunks_retrieved=0,
    )


# ---------------------------------------------------------------------------
# Anomaly alert replay endpoint (for testing)
# ---------------------------------------------------------------------------


@app.post("/api/v1/alerts/simulate", tags=["Testing"])
async def simulate_alert(
    request: Request,
    alert_type: str = "null_spike",
    table_name: str = "raw_orders",
    severity: str = "high",
    user: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """Inject a synthetic alert into the Redis queue (admin only)."""
    alert = {
        "alert_type": alert_type,
        "tenant_id": user["tenant_id"],
        "table_name": table_name,
        "severity": severity,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "simulated": True,
    }
    redis_client = request.app.state.redis
    await redis_client.lpush(settings.redis.alert_queue_key, json.dumps(alert))
    logger.info("Simulated alert injected", alert=alert)
    return {"status": "injected", "alert": alert}


# ---------------------------------------------------------------------------
# WebSocket — real-time DQ score streaming
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages WebSocket connections for real-time broadcasting."""

    def __init__(self) -> None:
        self._active: Dict[str, List[WebSocket]] = {}  # tenant_id → connections

    async def connect(self, ws: WebSocket, tenant_id: str) -> None:
        await ws.accept()
        self._active.setdefault(tenant_id, []).append(ws)
        logger.info(
            "WebSocket connected",
            tenant_id=tenant_id,
            connections=len(self._active[tenant_id]),
        )

    def disconnect(self, ws: WebSocket, tenant_id: str) -> None:
        if tenant_id in self._active:
            (
                self._active[tenant_id].discard(ws)
                if hasattr(self._active[tenant_id], "discard")
                else None
            )
            try:
                self._active[tenant_id].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, tenant_id: str, message: Dict[str, Any]) -> None:
        dead = []
        for ws in self._active.get(tenant_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, tenant_id)


ws_manager = ConnectionManager()


@app.websocket("/ws/dq-scores/{tenant_id}")
async def dq_scores_websocket(websocket: WebSocket, tenant_id: str) -> None:
    """
    Real-time WebSocket stream of DQ scores.
    Polls Redis every 5 seconds and broadcasts updates to connected clients.
    """
    await ws_manager.connect(websocket, tenant_id)
    redis_client = app.state.redis

    try:
        while True:
            # Fetch all DQ scores for the tenant from Redis
            pattern = f"{settings.redis.dq_scores_key}:{tenant_id}:*"
            keys = await redis_client.keys(pattern)
            scores = []
            for key in keys:
                raw = await redis_client.get(key)
                if raw:
                    scores.append(json.loads(raw))

            if scores:
                await websocket.send_json(
                    {
                        "type": "dq_scores_update",
                        "tenant_id": tenant_id,
                        "scores": scores,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, tenant_id)
        logger.info("WebSocket disconnected", tenant_id=tenant_id)


@app.websocket("/ws/alerts/{tenant_id}")
async def alerts_websocket(websocket: WebSocket, tenant_id: str) -> None:
    """
    Real-time WebSocket stream for new incident alerts.
    Listens to Redis alert queue and pushes to connected clients.
    """
    await ws_manager.connect(websocket, tenant_id)
    redis_client = app.state.redis

    try:
        while True:
            # Non-blocking pop from alert queue
            alert_raw = await redis_client.rpop(settings.redis.alert_queue_key)
            if alert_raw:
                alert = json.loads(alert_raw)
                if alert.get("tenant_id") == tenant_id:
                    await websocket.send_json(
                        {
                            "type": "new_alert",
                            "alert": alert,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, tenant_id)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "HTTP error",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception", error=str(exc), path=str(request.url), exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )
