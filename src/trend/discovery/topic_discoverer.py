import numpy as np
from models.models import ClusteringResults, DiscoveredTopic
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN


class TopicDiscoverer:
    def __init__(self, docs, years, vectors) -> None:
        self.docs = docs
        self.years = years
        self.embeddings = np.array([np.array(vector) for vector in vectors])

    def init_model(self):
        vectorizer_model = CountVectorizer(stop_words="english")
        umap_model = UMAP(n_neighbors=15, n_components=4,
                          min_dist=0.0, metric='cosine', random_state=42)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        hdbscan_model = HDBSCAN(min_cluster_size=10,
                                metric='euclidean', prediction_data=True)

        self.topic_model = BERTopic(min_topic_size=15, ctfidf_model=ctfidf_model, vectorizer_model=vectorizer_model,
                                    umap_model=umap_model, hdbscan_model=hdbscan_model)
        self.topic_model.fit(self.docs, self.embeddings)

        readable_topic_labels = []
        num_topics = len(self.topic_model.topic_labels_.keys()) + \
            (-1 if -1 in self.topic_model.topic_labels_ else 0)

        for i in range(0, min(10, num_topics)):
            readable_topic_labels.append(
                f"{i}. {', '.join(self.topic_model.topic_labels_[i].split('_')[1:])}")

        return readable_topic_labels

    def topics_over_time(self) -> list[DiscoveredTopic]:
        topics, _ = self.topic_model.transform(self.docs, self.embeddings)
        topics_over_time = self.topic_model.topics_over_time(
            self.docs, self.years)

        freq_df = self.topic_model.get_topic_freq()
        freq_df = freq_df.loc[freq_df.Topic != -1, :]
        if topics is not None:
            selected_topics = list()

        selected_topics = sorted(freq_df.Topic.to_list()[:10])

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
            id=i
        ) for i in selected_topics]

        return results

    def cluster_documents(self, sample: float = None) -> ClusteringResults:
        topic_per_doc = self.topic_model.topics_

        # Sample data if required
        if sample is not None and sample < 1:
            sampled_indices = []
            for topic in set(topic_per_doc):
                indices = np.where(np.array(topic_per_doc) == topic)[0]
                size = int(len(indices) * sample)
                sampled_indices.extend(np.random.choice(
                    indices, size=size, replace=False))
            sampled_topics = [topic_per_doc[i] for i in sampled_indices]
            if self.embeddings is not None:
                self.embeddings = self.embeddings[sampled_indices]
        else:
            sampled_topics = topic_per_doc

        umap_model = UMAP(n_neighbors=15, n_components=3,
                          min_dist=0.0, metric='cosine', random_state=42).fit(self.embeddings)
        embeddings_2d = umap_model.embedding_

        # Separate the coordinates and topic labels
        points_x = [float(coord[0]) for coord in embeddings_2d]
        points_y = [float(coord[1]) for coord in embeddings_2d]
        points_z = [float(coord[2]) for coord in embeddings_2d]
        topic_labels = sampled_topics

        return ClusteringResults(points_x, points_y, points_z, topic_labels)
