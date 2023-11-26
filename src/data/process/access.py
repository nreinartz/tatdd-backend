import asyncpg


async def prepare_database(conn: asyncpg.Connection):
    await create_table(conn)


async def create_table(conn: asyncpg.Connection):
    create_table_query = """
        CREATE TABLE IF NOT EXISTS queries (
            uuid TEXT PRIMARY KEY,
            type INTEGER NOT NULL,
            progress INTEGER NOT NULL,
            topics TEXT[],
            start_year INTEGER NOT NULL,
            end_year INTEGER NOT NULL,
            cutoff NUMERIC(4, 3) NOT NULL,
            min_citations INTEGER,
            results JSON
        );
    """
    await conn.execute(create_table_query)
