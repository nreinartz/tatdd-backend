import numpy as np
from models.models import ClusteringResults, DiscoveredTopic, TopicDiscoveryResults
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class TopicDiscoverer:
    def __init__(self, docs, years) -> None:
        self.docs = docs
        self.years = years

    def init_model(self):
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = sentence_model.encode(
            self.docs, show_progress_bar=False)

        umap_model = UMAP(n_neighbors=20, n_components=5,
                          min_dist=0.0, metric='cosine', random_state=42)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        vectorizer_model = CountVectorizer(stop_words="english")
        representation_model = MaximalMarginalRelevance(diversity=0.7)

        self.topic_model = BERTopic(min_topic_size=30, ctfidf_model=ctfidf_model, vectorizer_model=vectorizer_model,
                                    umap_model=umap_model, representation_model=representation_model)
        self.topic_model.fit(self.docs, self.embeddings)

    def discover_topics(self) -> TopicDiscoveryResults:
        topics, _ = self.topic_model.transform(self.docs, self.embeddings)
        topics_over_time = self.topic_model.topics_over_time(
            self.docs, self.years)

        freq_df = self.topic_model.get_topic_freq()
        freq_df = freq_df.loc[freq_df.Topic != -1, :]
        if topics is not None:
            selected_topics = list()

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
            id=i
        ) for i in selected_topics]

        clustering_results = self.cluster_documents()

        readable_topic_labels = []
        for i in range(0, 10):
            readable_topic_labels.append(
                f"{i}. {', '.join(self.topic_model.topic_labels_[i].split('_')[1:])}")

        return TopicDiscoveryResults(topics=readable_topic_labels, clusters=clustering_results, topics_over_time=results)

    def cluster_documents(self):
        return self.extract_topic_coordinates()

    def extract_topic_coordinates(self, sample: float = None) -> (list[(float, float, int)], list[(int, str)]):
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

        umap_model = UMAP(n_neighbors=10, n_components=3,
                          min_dist=0.0, metric='cosine').fit(self.embeddings)
        embeddings_2d = umap_model.embedding_

        # Separate the coordinates and topic labels
        points_x = [float(coord[0]) for coord in embeddings_2d]
        points_y = [float(coord[1]) for coord in embeddings_2d]
        points_z = [float(coord[2]) for coord in embeddings_2d]
        topic_labels = sampled_topics

        return ClusteringResults(points_x, points_y, points_z, topic_labels)
