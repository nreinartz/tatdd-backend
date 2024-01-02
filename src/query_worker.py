import numpy as np
from fastapi.concurrency import run_in_threadpool

from data.process.query_repository import QueryRepository
from data.weaviate.weaviate_data_provider import WeaviateAccessor

from models.models import AnalysisResults, CitationRecommendationResults, DataStatistics, Publication, QueryEntry, QueryProgress, QueryType, SearchResults, TopicDiscoveryResults, TrendResults

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
        except Exception as e:
            await query_repo.update_query_progress(entry.uuid, QueryProgress.FAILED)
            raise e

    if entry.type & QueryType.CITATION_RECOMMENDATION:
        try:
            await __fetch_citation_recommendations(query_repo, entry, weaviate_accessor)
        except Exception as e:
            await query_repo.update_query_progress(entry.uuid, QueryProgress.FAILED)
            raise e

    if entry.type & QueryType.TREND_ANALYSIS:
        try:
            await __discover_topics(query_repo, entry, weaviate_accessor)
        except Exception as e:
            await query_repo.update_query_progress(entry.uuid, QueryProgress.FAILED)
            raise e

    await query_repo.update_query_progress(entry.uuid, QueryProgress.FINISHED)


async def __fetch_data(query_repo: QueryRepository, entry: QueryEntry, weaviate_accessor: WeaviateAccessor, data_statistics: DataStatistics):
    num_pubs_found = 0
    adjusted_cutoff = entry.cutoff

    while num_pubs_found < 500:
        per_year = await run_in_threadpool(
            lambda: weaviate_accessor.get_publications_per_year(
                entry.topics, adjusted_cutoff, entry.start_year, entry.end_year)
        )
        num_pubs_found = sum(per_year.values())

        if num_pubs_found < 500:
            adjusted_cutoff = adjusted_cutoff - 0.01
            print("Adjusting cutoff to ", adjusted_cutoff)

    await query_repo.update_query_entry(entry)

    pub_objects = await run_in_threadpool(
        lambda: weaviate_accessor.get_publications_per_year_adjusted(
            entry.topics, data_statistics.publications_per_year, entry.start_year, entry.end_year)
    )

    year_value_pairs = {year: []
                        for year in range(entry.start_year, entry.end_year + 1)}
    pub_type_count = {}

    for pub_object in pub_objects:
        year_value_pairs[int(pub_object.properties["year"])].append(
            (1 - pub_object.metadata.distance))
        pub_type = pub_object.properties["type"].lower()
        pub_type_count[pub_type] = (
            pub_type_count[pub_type] if pub_type in pub_type_count else 0) + 1

    raw_values = [
        np.mean(year_value_pairs[year]) for year in range(entry.start_year, entry.end_year + 1)
    ]

    clamped_values = np.maximum(raw_values, adjusted_cutoff)

    if np.max(clamped_values) > np.min(clamped_values):
        adjusted_values = np.round(100 * (np.array(clamped_values) - np.min(clamped_values)) / (
            np.max(clamped_values) - np.min(clamped_values))).tolist()
    else:
        adjusted_values = [0 for _ in range(
            entry.start_year, entry.end_year + 1)]

    per_year_values = [per_year[year]
                       for year in range(entry.start_year, entry.end_year + 1)]

    entry.results.search_results = SearchResults(
        raw=raw_values,
        raw_per_year=per_year_values,
        adjusted=adjusted_values,
        pub_types=pub_type_count,
        adjusted_cutoff=adjusted_cutoff if adjusted_cutoff != entry.cutoff else None
    )

    # Return entry here since we updates properties
    return entry


async def __analyse_trends(query_repo: QueryRepository, entry: QueryEntry, trend_analyser: TrendAnalyser,
                           trend_descriptor: BaseTrendDescriptor, weaviate_accessor: WeaviateAccessor,
                           data_statistics: DataStatistics):

    entry.progress = QueryProgress.DATA_RETRIEVAL
    await query_repo.update_query_entry(entry)

    try:
        entry = await __fetch_data(query_repo, entry, weaviate_accessor, data_statistics)
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
    await query_repo.update_query_progress(entry.uuid, QueryProgress.CLUSTERING_TOPICS)

    max_documents = 6500

    matching_pubs = weaviate_accessor.get_matching_publications_with_vector(
        entry.topics, entry.start_year, entry.end_year, max_documents
    )

    docs = [
        f"{x.properties['title']}: {x.properties['abstract']}" for x in matching_pubs]
    years = [x.properties["year"] for x in matching_pubs]
    vectors = [x.vector for x in matching_pubs]

    topic_discoverer = TopicDiscoverer(docs, years, vectors)

    topics = await run_in_threadpool(
        lambda: topic_discoverer.init_model()
    )

    discovery_results = TopicDiscoveryResults(
        topics, None, None)

    entry.results.topic_discovery_results = discovery_results

    discovery_results.clusters = await run_in_threadpool(
        lambda: topic_discoverer.cluster_documents()
    )

    entry.progress = QueryProgress.TOPICS_OVER_TIME
    await query_repo.update_query_entry(entry)

    discovery_results.topics_over_time = await run_in_threadpool(
        lambda: topic_discoverer.topics_over_time()
    )

    await query_repo.update_query_entry(entry)


async def __fetch_citation_recommendations(query_repo: QueryRepository, entry: QueryEntry, weaviate_accessor: WeaviateAccessor):
    await query_repo.update_query_progress(entry.uuid, QueryProgress.CITATION_RETRIEVAL)

    publications = await run_in_threadpool(
        lambda: weaviate_accessor.get_matching_publications(
            entry.topics, entry.start_year, entry.end_year, 20, entry.min_citations)
    )

    entry.results.citation_results = CitationRecommendationResults(
        publications=[Publication(
            title=x.properties["title"],
            doi=x.properties["doi"],
            authors=x.properties["authors"],
            year=x.properties["year"],
            type=x.properties["type"],
            abstract=x.properties["abstract"],
            similarity=1-x.metadata.distance,
            citations=x.properties["n_citations"] if "n_citations" in x.properties else None
        ) for x in publications]
    )

    await query_repo.update_query_entry(entry)
