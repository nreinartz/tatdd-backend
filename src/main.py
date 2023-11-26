from contextlib import asynccontextmanager
import datetime
import asyncpg
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from dotenv import load_dotenv
from dataclasses import asdict
from fastapi import Depends, FastAPI, BackgroundTasks, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import weaviate

from models.models import DataStatistics, QueryEntry, QueryRequest
from query_worker import process_query
from data.process.access import prepare_database
from data.process.query_repository import QueryRepository
from data.weaviate.weaviate_data_provider import WeaviateAccessor
from trend.analysis.trend_analyser import TrendAnalyser, get_trend_analyser
from trend.chart.chart_generator import generate_trend_chart

# Trend description
from trend.descriptor.base_descriptor import BaseTrendDescriptor
from trend.descriptor.gpt_descriptor import get_gpt_descriptor
from trend.descriptor.rule_based_descriptor import get_rule_based_descriptor

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "trend_api")
CONNECTION_STRING = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")
WEAVIATE_REST_PORT = os.getenv("WEAVIATE_PORT", "8080")
WEAVIATE_GRPC_PORT = os.getenv("WEAVIATE_GRPC_PORT", "50051")
WEAVIATE_ENDPOINT = f"http://{WEAVIATE_HOST}:{WEAVIATE_REST_PORT}"

TRENDDESCRIPTOR = os.getenv("TREND_DESCRIPTOR", "rule_based")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.pool = await asyncpg.create_pool(CONNECTION_STRING)
    app.state.weaviate_client = weaviate.WeaviateClient(
        weaviate.ConnectionParams.from_url(
            WEAVIATE_ENDPOINT, WEAVIATE_GRPC_PORT)
    )

    async with app.state.pool.acquire() as connection:
        await prepare_database(connection)

    # Initial blocking run at startup instead of next_run_time, since we cannot work without it
    update_data_statistics()

    scheduler.add_job(
        update_data_statistics,
        trigger=IntervalTrigger(minutes=10)
    )
    scheduler.start()

    yield

    # Shutdown
    await app.state.pool.close()
    scheduler.shutdown()

scheduler = AsyncIOScheduler()
app = FastAPI(openapi_url="/swagger.json", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=[
                   "*"], allow_methods=["*"], allow_headers=["*"])


# Dependency Injection
async def get_query_repository() -> QueryRepository:
    async with app.state.pool.acquire() as connection:
        try:
            yield QueryRepository(connection)
        finally:
            await connection.close()


def get_trend_descriptor() -> BaseTrendDescriptor:
    if TRENDDESCRIPTOR == "gpt":
        return get_gpt_descriptor()
    else:
        return get_rule_based_descriptor()


def get_weaviate_accessor() -> WeaviateAccessor:
    return WeaviateAccessor(app.state.weaviate_client)


def update_data_statistics():
    print("Fetching data statistics ...")

    accessor = get_weaviate_accessor()

    pubs_per_year = dict()
    for year in range(1980, datetime.datetime.now().year + 1):
        pubs_per_year[year] = accessor.get_statistics_for_year(year)

    app.state.data_statistics = DataStatistics(
        total_publications=sum(pubs_per_year.values()),
        publications_per_year=pubs_per_year
    )

    print("Done fetching data statistics, total publications: {}".format(
        app.state.data_statistics.total_publications))


@app.post("/api/queries", response_model=QueryEntry, status_code=status.HTTP_201_CREATED)
async def create_process_request(query_request: QueryRequest, background_tasks: BackgroundTasks, query_repo: QueryRepository = Depends(get_query_repository),
                                 weaviate_accessor: WeaviateAccessor = Depends(get_weaviate_accessor), trend_analyser: TrendAnalyser = Depends(get_trend_analyser),
                                 trend_descriptor: BaseTrendDescriptor = Depends(get_trend_descriptor)):

    query_request.cutoff = max(0.7, min(0.98, query_request.cutoff))
    entry: QueryEntry = await query_repo.create_query_entry(query_request)

    background_tasks.add_task(
        process_query,
        entry.uuid,
        query_repo,
        weaviate_accessor,
        trend_analyser,
        trend_descriptor,
        app.state.data_statistics
    )

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=asdict(entry))


@app.get("/api/statistics", response_model=DataStatistics, status_code=status.HTTP_200_OK)
async def get_data_statistics():
    return JSONResponse(status_code=status.HTTP_200_OK, content=asdict(app.state.data_statistics))


@app.get("/api/queries/{query_id}", response_model=QueryEntry)
async def get_process_progress(query_id: str, query_repo: QueryRepository = Depends(get_query_repository)):
    entry = await query_repo.get_query_entry(query_id)
    if entry is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": "Query not found"})
    return JSONResponse(status_code=status.HTTP_200_OK, content=asdict(entry))


@app.get("/api/queries/{query_id}/summary", response_model=QueryEntry)
async def get_process_progress(query_id: str, query_repo: QueryRepository = Depends(get_query_repository)):
    entry = await query_repo.get_query_summary(query_id)
    if entry is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": "Query not found"})
    return JSONResponse(status_code=status.HTTP_200_OK, content=asdict(entry))


@app.head("/api/queries/{query_id}/chart")
@app.get("/api/queries/{query_id}/chart")
async def get_process_progress(query_id: str, query_repo: QueryRepository = Depends(get_query_repository), format: str = "svg"):
    entry = await query_repo.get_query_entry(query_id)
    if entry is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": "Query not found"})

    content = generate_trend_chart(entry, format)

    headers = {
        "Content-Type": "image/png" if format == "png" else "image/svg+xml",
        "Cache-Control": "public,max-age=3600",
        "Accept-Ranges": "bytes",
        "etag": "1",
    }

    return Response(content=content, headers=headers, media_type=headers["Content-Type"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="127.0.0.1", port=8000)
