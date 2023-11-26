import dataclasses
import json
import uuid

from asyncpg import Connection
from models.models import QueryEntry, QueryProgress, QueryRequest


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


class QueryRepository:
    def __init__(self, conn: Connection):
        self.conn = conn

    async def create_query_entry(self, entry: QueryRequest) -> QueryEntry:
        entry = QueryEntry(uuid=str(uuid.uuid4()), type=entry.query_type, progress=QueryProgress.QUEUED, topics=entry.topics,
                           start_year=entry.start_year, end_year=entry.end_year, cutoff=entry.cutoff, min_citations=entry.min_citations, results=None)
        insert_query = "INSERT INTO queries (uuid, type, progress, topics, start_year, end_year, cutoff, min_citations, results) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);"
        await self.conn.execute(insert_query, *dataclasses.astuple(entry))
        return entry

    async def get_query_entry(self, uuid: str) -> QueryEntry:
        select_query = "SELECT * FROM queries WHERE uuid = $1;"
        row = await self.conn.fetchrow(select_query, uuid)
        if row == None:
            return None
        results_map = {**{i: row[i] for i in row.keys() if i != 'results'},
                       "cutoff": float(row["cutoff"]),
                       "results": json.loads(row["results"]) if row["results"] != None else None}
        return QueryEntry(**results_map)

    async def get_query_summary(self, uuid: str) -> QueryEntry:
        select_query = "SELECT uuid, type, progress, topics, start_year, end_year, cutoff, min_citations FROM queries WHERE uuid = $1;"
        row = await self.conn.fetchrow(select_query, uuid)
        if row == None:
            return None
        results_map = {**{i: row[i] for i in row.keys() if i != 'results'},
                       "cutoff": float(row["cutoff"]),
                       "results":  None}
        return QueryEntry(**results_map)

    async def update_query_entry(self, entry: QueryEntry):
        update_query = "UPDATE queries SET progress = $1, results = $2 WHERE uuid = $3;"
        await self.conn.execute(update_query, entry.progress, json.dumps(entry.results, cls=EnhancedJSONEncoder), entry.uuid)

    async def update_query_progress(self, uuid: str, progress: QueryProgress):
        update_query = "UPDATE queries SET progress = $1 WHERE uuid = $2;"
        await self.conn.execute(update_query, progress, uuid)

    async def get_all_query_entries(self) -> list[QueryEntry]:
        select_query = "SELECT * FROM queries;"
        rows = await self.conn.fetch(select_query)
        return [QueryEntry(**row) for row in rows]

    async def delete_query_entry(self, uuid: str):
        delete_query = "DELETE FROM queries WHERE uuid = $1;"
        await self.conn.execute(delete_query, uuid)
