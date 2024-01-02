import weaviate
from weaviate.classes import Filter, MetadataQuery

import datetime
import numpy as np


class WeaviateAccessor:
    def __init__(self, client: weaviate.WeaviateClient):
        self.client = client
        self.publications = self.client.collections.get("Publication")

    def get_grouped_per_year(self, concepts: list[str],
                             cutoff: float, group_prop: str, start_year: int = 1000,
                             end_year: int = datetime.datetime.now().year):

        return self.publications.aggregate_group_by.near_text(
            query=concepts,
            distance=1 - cutoff,
            filters=Filter("year").greater_or_equal(
                start_year) & Filter("year").less_or_equal(end_year),
            group_by=group_prop
        )

    def get_publications_in_year(self, concepts: list[str], year: int, limit: int = 2000):
        results = self.publications.query.near_text(
            query=concepts,
            filters=Filter("year").equal(year),
            include_vector=True,
            return_properties=["title", "abstract", "year"],
            limit=limit
        )

        return results.objects

    def get_publications_per_year(self, concepts: list[str],
                                  cutoff: float, start_year: int = 1000,
                                  end_year: int = datetime.datetime.now().year):
        query_results = self.get_grouped_per_year(
            concepts, cutoff, "year", start_year, end_year)

        results_default = {
            key: 0
            for key in range(start_year, end_year + 1)
        }

        results_query = dict(
            [(int(entry.grouped_by.value), entry.total_count)
             for entry in query_results]
        )

        return {**results_default, **results_query}

    def get_publications_per_year_adjusted(self, concepts: list[str], year_stats: dict[int, int],
                                           start_year: int = 1000, end_year: int = datetime.datetime.now().year):
        objects = []
        for i in range(start_year, end_year + 1):
            year_result = self.publications.query.near_text(
                query=concepts,
                filters=Filter("year").equal(i),
                return_properties=["year", "type"],
                return_metadata=MetadataQuery(distance=True),
                limit=int(np.log10(year_stats[i])*10)
            )
            objects += year_result.objects

        return objects

    def get_count_per_pub_type(self, concepts: list[str],
                               cutoff: float, start_year: int = 1000,
                               end_year: int = datetime.datetime.now().year):
        query_results = self.get_grouped_per_year(
            concepts, cutoff, "type", start_year, end_year)

        return dict(
            [(entry.grouped_by.value, entry.total_count)
             for entry in query_results]
        )

    def get_matching_publications(self, concepts: list[str], start_year: int,
                                  end_year: int, limit: int = 3000, min_citation_count: int | None = None):

        filters = Filter("year").greater_or_equal(
            start_year) & Filter("year").less_or_equal(end_year)

        if min_citation_count != None and min_citation_count > 0:
            filters = filters & Filter(
                "n_citations").greater_or_equal(min_citation_count)

        results = self.publications.query.near_text(
            query=concepts,
            filters=filters,
            return_properties=["title", "doi", "authors",
                               "year", "type", "abstract", "n_citations"],
            return_metadata=MetadataQuery(distance=True),
            limit=limit
        )

        return results.objects

    def get_matching_publications_with_vector(self, concepts: list[str], start_year: int,
                                              end_year: int, limit: int = 3000):

        filters = Filter("year").greater_or_equal(
            start_year) & Filter("year").less_or_equal(end_year)

        results = self.publications.query.near_text(
            query=concepts,
            filters=filters,
            include_vector=True,
            return_properties=["title", "abstract", "year"],
            limit=limit
        )

        return results.objects

    def get_statistics_for_year(self, year: int) -> int:
        results = self.publications.aggregate_group_by.over_all(
            filters=Filter("year").equal(year),
            group_by="year"
        )

        return results[0].total_count
