import weaviate
import datetime


class WeaviateAccessor:
    def __init__(self, client: weaviate.client.Client):
        self.client = client

    def get_grouped_per_year(self, concepts: list[str],
                             distance: float, group_prop: str, start_year: int = 1000,
                             end_year: int = datetime.datetime.now().year) -> int:
        return (self.client.query.aggregate("Publication")
                .with_fields("meta { count } groupedBy { value }")
                .with_near_text({"concepts": concepts, "distance": distance})
                .with_meta_count()
                .with_where(self.__construct_year_filter(start_year, end_year))
                .with_group_by_filter([group_prop])
                .do())

    def get_publications_in_year(self, concepts: list[str], year: int, limit: int = 2000):
        results = (self.client.query.get("Publication", ["title", "abstract", "publication_year"])
                   .with_near_text({"concepts": concepts})
                   .with_additional(["vector"])
                   .with_where({"path": "publication_year", "operator": "Equal", "valueInt": year})
                   .with_limit(limit)
                   .do())

        if "data" not in results:
            return []

        return results['data']['Get']['Publication']

    def get_publications_per_year(self, concepts: list[str],
                                  distance: float, start_year: int = 1000,
                                  end_year: int = datetime.datetime.now().year) -> int:
        query_results = self.get_grouped_per_year(
            concepts, distance, "publication_year", start_year, end_year)

        results_default = {
            key: 0
            for key in range(start_year, end_year)
        }

        results_query = dict(
            [(int(x["groupedBy"]["value"]), x["meta"]["count"])
             for x in query_results["data"]["Aggregate"]["Publication"]]
        )

        return {**results_default, **results_query}

    def get_count_per_pub_type(self, concepts: list[str],
                               distance: float, start_year: int = 1000,
                               end_year: int = datetime.datetime.now().year) -> int:
        query_results = self.get_grouped_per_year(
            concepts, distance, "publication_type", start_year, end_year)
        return dict(
            [(x["groupedBy"]["value"], x["meta"]["count"])
             for x in query_results["data"]["Aggregate"]["Publication"]]
        )

    def get_matching_publications(self, concepts: list[str], start_year: int,
                                  end_year: int, limit: int = 3000, min_citation_count: int | None = None) -> list[dict]:
        query = (self.client.query
                 .get("Publication", ["title", "abstract", "publication_year", "doi", "publication_type", "authors", "n_citations"])
                 .with_additional("distance")
                 .with_near_text({"concepts": concepts})
                 .with_limit(limit))

        if start_year != None or end_year != None:
            query = query.with_where(
                self.__construct_year_filter(start_year, end_year))

        if min_citation_count != None:
            query = query.with_where({
                "path": ["n_citations"],
                "operator": "GreaterThanEqual",
                "valueInt": min_citation_count
            })

        response = query.do()

        return response["data"]["Get"]["Publication"][1:]

    def get_statistics_for_year(self, year: int) -> int:
        return (self.client.query.aggregate("Publication")
                .with_where({"path": ["publication_year"], "operator": "Equal", "valueInt": year})
                .with_meta_count()
                .do()
                )["data"]["Aggregate"]["Publication"][0]["meta"]["count"]

    def __construct_year_filter(self, start_year: int = None, end_year: int = None) -> dict:
        return {
            "operator": "And",
            "operands": [
                {
                    "path": ["publication_year"],
                    "operator": "GreaterThanEqual",
                    "valueInt": start_year,
                }, {
                    "path": ["publication_year"],
                    "operator": "LessThanEqual",
                    "valueInt": end_year
                }
            ]
        }
