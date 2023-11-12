import numpy as np
from models.models import DiscoveredTopic, TopicDiscoveryResults
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def get_topic_discoverer():
    return TopicDiscoverer()


class TopicDiscoverer:
    def discover_topics(self, documents: list[str], years: list[int], vectors) -> TopicDiscoveryResults:
        local_vectors = np.array([np.array(vector) for vector in vectors])

        vectorizer_model = CountVectorizer(stop_words="english")

        umap_model = UMAP(n_neighbors=20, n_components=5,
                          min_dist=0.0, metric='cosine', random_state=42)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        pipe = make_pipeline(
            TfidfVectorizer(),
            TruncatedSVD(100)
        )

        representation_model = MaximalMarginalRelevance(diversity=0.7)
        vectorizer_model = CountVectorizer(stop_words="english")

        topic_model = BERTopic(verbose=True, embedding_model=pipe, min_topic_size=50, ctfidf_model=ctfidf_model,
                               vectorizer_model=vectorizer_model, umap_model=umap_model, representation_model=representation_model)
        topics, _ = topic_model.fit_transform(documents)
        topics_over_time = topic_model.topics_over_time(documents, years)

        freq_df = topic_model.get_topic_freq()
        freq_df = freq_df.loc[freq_df.Topic != -1, :]
        if topics is not None:
            selected_topics = list(topics)

        selected_topics = sorted(freq_df.Topic.to_list()[:10])
        selected_topics

        df = topics_over_time.loc[topics_over_time.Topic.isin(
            selected_topics), :]
        df = df.groupby("Topic")

        grouped_dict = df.agg(lambda x: list(x))[
            ["Words", "Frequency", "Timestamp"]].to_dict()

        results = [DiscoveredTopic(
            words=[word_str.split(", ")
                   for word_str in grouped_dict["Words"][i]],
            frequencies=grouped_dict["Frequency"][i],
            timestamps=grouped_dict["Timestamp"][i],
            name=topic_model.topic_labels_[i]
        ) for i in selected_topics]

        return TopicDiscoveryResults(topics=results)
