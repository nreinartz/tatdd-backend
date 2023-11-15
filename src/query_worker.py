import numpy as np
from fastapi.concurrency import run_in_threadpool

from data.process.query_repository import QueryRepository
from data.weaviate.weaviate_data_provider import WeaviateAccessor

from models.models import AnalysisResults, CitationRecommendationResults, DataStatistics, Publication, QueryEntry, QueryProgress, QueryType, SearchResults, TrendResults

from trend.descriptor.base_descriptor import BaseTrendDescriptor
from trend.analysis.trend_analyser import TrendAnalyser
from trend.discovery.topic_discoverer import TopicDiscoverer


async def process_query(uuid: str, query_repo: QueryRepository,
                        weaviate_accessor: WeaviateAccessor, trend_analyser: TrendAnalyser,
                        trend_descriptor: BaseTrendDescriptor, data_statistics: DataStatistics):

    entry = await query_repo.get_query_entry(uuid)
    entry.results = AnalysisResults()

    if entry.type & QueryType.TREND_ANALYSIS:
        try:
            await __analyse_trends(query_repo, entry, trend_analyser,
                                   trend_descriptor, weaviate_accessor, data_statistics)

            await __discover_topics(query_repo, entry, weaviate_accessor)
        except Exception as e:
            await query_repo.update_query_progress(entry.uuid, QueryProgress.FAILED)
            raise e

    if entry.type & QueryType.CITATION_RECOMMENDATION:
        try:
            await __fetch_citation_recommendations(query_repo, entry, weaviate_accessor)
        except Exception as e:
            await query_repo.update_query_progress(entry.uuid, QueryProgress.FAILED)
            raise e

    await query_repo.update_query_progress(entry.uuid, QueryProgress.FINISHED)


async def __fetch_data(entry: QueryEntry, weaviate_accessor: WeaviateAccessor, data_statistics: DataStatistics):
    results = await run_in_threadpool(
        lambda: weaviate_accessor.get_publications_per_year(
            entry.topics, entry.distance, entry.start_year, entry.end_year)
    )

    pub_types = await run_in_threadpool(
        lambda: weaviate_accessor.get_count_per_pub_type(
            entry.topics, entry.distance, entry.start_year, entry.end_year)
    )
    raw_values = [
        results[key] for key in sorted(list(results.keys()))
    ]
    adjusted_values = [
        (results[key] / data_statistics.publications_per_year[key]) * 100
        for key in sorted(list(results.keys()))
    ]
    adjusted_values = np.round(
        (adjusted_values/np.max(adjusted_values))*100).tolist()

    return SearchResults(raw=raw_values, adjusted=adjusted_values, pub_types=pub_types)


async def __analyse_trends(query_repo: QueryRepository, entry: QueryEntry, trend_analyser: TrendAnalyser,
                           trend_descriptor: BaseTrendDescriptor, weaviate_accessor: WeaviateAccessor,
                           data_statistics: DataStatistics):

    entry.progress = QueryProgress.DATA_RETRIEVAL
    await query_repo.update_query_entry(entry)

    try:
        entry.results.search_results = await __fetch_data(entry, weaviate_accessor, data_statistics)
    except Exception as e:
        await query_repo.update_query_progress(entry.uuid, QueryProgress.FAILED)
        raise e

    entry.progress = QueryProgress.ANALYSING_TRENDS
    await query_repo.update_query_entry(entry)

    years = list(range(entry.start_year, entry.end_year + 1))
    breakpoints, trends = await run_in_threadpool(lambda: trend_analyser.analyse(years, entry.results.search_results.adjusted))

    entry.results.trend_results = TrendResults(
        breakpoints=breakpoints,
        global_trend=trends[0],
        sub_trends=trends[1:]
    )
    entry.progress = QueryProgress.GENERATING_DESCRIPTION
    await query_repo.update_query_entry(entry)

    entry.results.trend_results.trend_description = await run_in_threadpool(
        lambda: trend_descriptor.generate_description(
            entry.topics,
            entry.start_year,
            entry.end_year,
            entry.results.search_results.adjusted,
            entry.results.trend_results.global_trend,
            entry.results.trend_results.sub_trends
        )
    )

    await query_repo.update_query_entry(entry)


async def __discover_topics(query_repo: QueryRepository, entry: QueryEntry, weaviate_accessor: WeaviateAccessor):
    await query_repo.update_query_progress(entry.uuid, QueryProgress.DISCOVERING_TOPICS)

    max_documents = 6000
    pop_sum = sum(entry.results.search_results.adjusted)

    docs = []
    years = []

    for year in range(entry.start_year, entry.end_year + 1):
        if entry.results.search_results.adjusted[year - entry.start_year] == 0:
            continue

        doc_limit = int(np.round(
            max_documents * (entry.results.search_results.adjusted[year - entry.start_year] / pop_sum)))

        if doc_limit == 0:
            continue

        publications = await run_in_threadpool(
            lambda: weaviate_accessor.get_publications_in_year(
                entry.topics, year, doc_limit)
        )
        docs.extend(
            [f"{x.properties['title']}: {x.properties['abstract']}" for x in publications])

        years.extend([x.properties["year"] for x in publications])

    topic_discoverer = TopicDiscoverer(docs, years)

    await run_in_threadpool(
        lambda: topic_discoverer.init_model()
    )

    entry.results.topic_discovery_results = await run_in_threadpool(
        lambda: topic_discoverer.discover_topics()
    )

    await query_repo.update_query_entry(entry)


async def __fetch_citation_recommendations(query_repo: QueryRepository, entry: QueryEntry, weaviate_accessor: WeaviateAccessor):
    await query_repo.update_query_progress(entry.uuid, QueryProgress.CITATION_RETRIEVAL)

    publications = await run_in_threadpool(
        lambda: weaviate_accessor.get_matching_publications(
            entry.topics, entry.start_year, entry.end_year, 15, entry.min_citations)
    )

    entry.results.citation_results = CitationRecommendationResults(
        publications=[Publication(
            title=x.properties["title"],
            doi=x.properties["doi"],
            authors=x.properties["authors"],
            year=x.properties["year"],
            type=x.properties["type"],
            abstract=x.properties["abstract"],
            distance=x.metadata.distance,
            citations=x.properties["n_citations"] if "n_citations" in x.properties else None
        ) for x in publications]
    )

    await query_repo.update_query_entry(entry)
