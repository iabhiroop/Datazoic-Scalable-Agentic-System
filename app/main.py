"""FastAPI process resources and normalized errors. The worker remains a separate process."""

from contextlib import AsyncExitStack, asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException

from app.agent.gateway import LLMError, create_llm_gateway
from app.agent.llm import ModelClient
from app.api.routes import router
from app.api.runs import RunService
from app.db.session import Database, event_loop_factory
from app.observability import TraceContext, initialize_telemetry
from app.security import AuthorizationService, TokenAuthenticator
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with AsyncExitStack() as resources:
            telemetry = await initialize_telemetry(settings)
            resources.push_async_callback(telemetry.aclose)
            application.state.telemetry = telemetry
            authenticator = TokenAuthenticator(settings)
            llm_gateway = None
            if (
                settings.llm_provider != "gemini"
                or settings.gemini_model_name
                or (settings.gemini_api_key and settings.gemini_api_key.get_secret_value())
            ):
                llm_gateway = create_llm_gateway(settings, telemetry=telemetry)
                resources.push_async_callback(llm_gateway.aclose)
            application.state.llm_gateway = llm_gateway
            application.state.model_client = ModelClient(llm_gateway) if llm_gateway else None
            database = None
            if settings.database_url is not None:
                database = Database(settings)
                resources.push_async_callback(database.close)
                await database.open()
            application.state.database = database
            application.state.authenticator = authenticator
            application.state.http_client = await resources.enter_async_context(
                httpx.AsyncClient(timeout=settings.tool_timeout_seconds)
            )
            application.state.run_service = RunService(
                database, settings, AuthorizationService(authenticator, database)
            )
            # Optional Redis/object storage clients are wired with their later slices.
            yield

    application = FastAPI(title="Simplified Agentic System", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def correlate_request(request: Request, call_next):
        request.state.request_id = str(uuid4())
        telemetry = request.app.state.telemetry
        with (
            telemetry.bind_context(TraceContext(request_id=request.state.request_id)),
            telemetry.observe("http.request", metadata={"method": request.method}) as observation,
        ):
            request.state.trace_context = telemetry.trace_context()
            response = await call_next(request)
            route = request.scope.get("route")
            observation.update(
                metadata={
                    "route": getattr(route, "path", "unmatched"),
                    "status_code": response.status_code,
                }
            )
            if response.status_code >= 400:
                observation.error(f"http_{response.status_code}")
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def error_response(request, status, code, message, details=None, headers=None):
        return JSONResponse(
            status_code=status,
            headers=headers,
            content={
                "code": code,
                "message": message,
                "request_id": request.state.request_id,
                "details": details or {},
            },
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        codes = {
            401: "unauthenticated",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            503: "unavailable",
        }
        return error_response(
            request,
            error.status_code,
            codes.get(error.status_code, "request_error"),
            str(error.detail),
            headers=error.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Never echo raw input, tokens or unrestricted validator context.
        violations = [
            {"location": list(item["loc"]), "type": item["type"]} for item in error.errors()
        ]
        return error_response(
            request, 422, "invalid_request", "Request validation failed", {"violations": violations}
        )

    @application.exception_handler(NotImplementedError)
    async def stub_error(request: Request, error: NotImplementedError):
        return error_response(request, 501, "not_implemented", str(error))

    @application.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, error: SQLAlchemyError):
        return error_response(request, 503, "unavailable", "Database operation unavailable")

    @application.exception_handler(LLMError)
    async def model_error(request: Request, error: LLMError):
        return error_response(request, 503 if error.retryable else 502, error.code, str(error))

    application.include_router(router)
    return application


app = create_app()


def main() -> None:
    """Local development entry point, including Windows/Psycopg compatibility."""
    import asyncio

    import uvicorn

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000)).serve())


if __name__ == "__main__":
    main()
