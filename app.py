# ============================================================
# MAI511-2 - ADVANCED MACHINE LEARNING
# LAB 7
# K-MEANS AND K-MEDOIDS CLUSTERING
# Streamlit Application
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from sklearn_extra.cluster import KMedoids


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="K-Means vs K-Medoids",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📊 K-Means vs K-Medoids Clustering")

st.subheader(
    "MAI511-2 – Advanced Machine Learning | Lab Exercise 7"
)

st.write(
    "Implementation and comparison of K-Means and K-Medoids "
    "clustering using the Iris dataset."
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(
    X,
    columns=feature_names
)

df["Species"] = [
    target_names[i] for i in y
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "K-Means",
        "K-Medoids",
        "Compare Both"
    ]
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=6,
    value=3
)

st.sidebar.info(
    "The Iris dataset contains 3 naturally occurring "
    "species. K = 3 is therefore used as the main "
    "cluster configuration."
)


# ============================================================
# FEATURE SCALING
# ============================================================

features = df[feature_names]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(features)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Number of Samples",
        df.shape[0]
    )

with col2:
    st.metric(
        "Number of Features",
        len(feature_names)
    )

with col3:
    st.metric(
        "Selected K",
        k
    )


st.write("### Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# PREPROCESSING
# ============================================================

st.header("2. Data Preprocessing")

st.write(
    "The species/target column is excluded from clustering. "
    "Only numerical features are used."
)

missing_values = df.isnull().sum().sum()

if missing_values == 0:
    st.success("No missing values found.")
else:
    st.warning(
        "Missing values found: "
        + str(missing_values)
    )


st.write("### Features Used")

st.write(feature_names)

st.write("### Scaled Data")

scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names
)

st.dataframe(
    scaled_df.head(10),
    use_container_width=True
)


# ============================================================
# ELBOW METHOD
# ============================================================

st.header("3. Elbow Method")

K_values = range(2, 11)

inertia = []

for current_k in K_values:

    model = KMeans(
        n_clusters=current_k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(
        model.inertia_
    )


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    list(K_values),
    inertia,
    marker="o"
)

ax.set_xlabel(
    "Number of Clusters (K)"
)

ax.set_ylabel(
    "Inertia"
)

ax.set_title(
    "Elbow Method"
)

ax.set_xticks(
    list(K_values)
)

ax.grid(True)

st.pyplot(fig)


st.info(
    "The elbow point is around K = 3 for the Iris dataset."
)


# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

st.header("4. Silhouette Score Analysis")

silhouette_values = []

for current_k in K_values:

    model = KMeans(
        n_clusters=current_k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_values.append(score)


silhouette_df = pd.DataFrame({
    "K": list(K_values),
    "Silhouette Score": silhouette_values
})


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    list(K_values),
    silhouette_values,
    marker="o"
)

ax.set_xlabel(
    "Number of Clusters (K)"
)

ax.set_ylabel(
    "Silhouette Score"
)

ax.set_title(
    "Silhouette Score vs Number of Clusters"
)

ax.set_xticks(
    list(K_values)
)

ax.grid(True)

st.pyplot(fig)

st.dataframe(
    silhouette_df,
    use_container_width=True
)


# ============================================================
# PCA
# ============================================================

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


# ============================================================
# FUNCTION - K-MEANS
# ============================================================

def run_kmeans(k):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    centroids = model.cluster_centers_

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    db = davies_bouldin_score(
        X_scaled,
        labels
    )

    ch = calinski_harabasz_score(
        X_scaled,
        labels
    )

    return (
        model,
        labels,
        centroids,
        silhouette,
        db,
        ch
    )


# ============================================================
# FUNCTION - K-MEDOIDS
# ============================================================

def run_kmedoids(k):

    model = KMedoids(
        n_clusters=k,
        metric="euclidean",
        random_state=42
    )

    labels = model.fit_predict(
        X_scaled
    )

    medoid_indices = model.medoid_indices_

    medoids = X_scaled[
        medoid_indices
    ]

    silhouette = silhouette_score(
        X_scaled,
        labels
    )

    db = davies_bouldin_score(
        X_scaled,
        labels
    )

    ch = calinski_harabasz_score(
        X_scaled,
        labels
    )

    return (
        model,
        labels,
        medoid_indices,
        medoids,
        silhouette,
        db,
        ch
    )


# ============================================================
# K-MEANS SECTION
# ============================================================

if algorithm == "K-Means":

    st.header("5. K-Means Clustering")

    (
        kmeans,
        labels,
        centroids,
        silhouette,
        db,
        ch
    ) = run_kmeans(k)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Silhouette Score",
            round(silhouette, 4)
        )

    with col2:
        st.metric(
            "Davies-Bouldin Index",
            round(db, 4)
        )

    with col3:
        st.metric(
            "Calinski-Harabasz",
            round(ch, 4)
        )


    # --------------------------------------------------------
    # PCA VISUALIZATION
    # --------------------------------------------------------

    st.subheader(
        "K-Means Cluster Visualization"
    )

    centroids_pca = pca.transform(
        centroids
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="viridis",
        s=60
    )

    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker="X",
        s=250,
        c="red",
        label="Centroids"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "K-Means Clustering using PCA"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)


    # --------------------------------------------------------
    # CENTROIDS
    # --------------------------------------------------------

    st.subheader(
        "K-Means Centroids"
    )

    centroids_original = scaler.inverse_transform(
        centroids
    )

    centroid_df = pd.DataFrame(
        centroids_original,
        columns=feature_names
    )

    centroid_df.index = [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3",
        "Cluster 4",
        "Cluster 5"
    ][:k]

    st.dataframe(
        centroid_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CLUSTER SIZE
    # --------------------------------------------------------

    st.subheader(
        "Cluster Distribution"
    )

    cluster_counts = pd.Series(
        labels
    ).value_counts().sort_index()

    cluster_counts.index = [
        "Cluster " + str(i)
        for i in cluster_counts.index
    ]

    st.bar_chart(
        cluster_counts
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "K-Means Interpretation"
    )

    st.write(
        "K-Means assigns each observation to the nearest "
        "centroid. The centroid represents the mean position "
        "of all observations belonging to that cluster."
    )


# ============================================================
# K-MEDOIDS SECTION
# ============================================================

elif algorithm == "K-Medoids":

    st.header("6. K-Medoids Clustering")

    (
        kmedoids,
        labels,
        medoid_indices,
        medoids,
        silhouette,
        db,
        ch
    ) = run_kmedoids(k)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Silhouette Score",
            round(silhouette, 4)
        )

    with col2:
        st.metric(
            "Davies-Bouldin Index",
            round(db, 4)
        )

    with col3:
        st.metric(
            "Calinski-Harabasz",
            round(ch, 4)
        )


    # --------------------------------------------------------
    # PCA VISUALIZATION
    # --------------------------------------------------------

    st.subheader(
        "K-Medoids Cluster Visualization"
    )

    medoids_pca = pca.transform(
        medoids
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="plasma",
        s=60
    )

    ax.scatter(
        medoids_pca[:, 0],
        medoids_pca[:, 1],
        marker="X",
        s=250,
        c="red",
        label="Medoids"
    )

    ax.set_xlabel(
        "Principal Component 1"
    )

    ax.set_ylabel(
        "Principal Component 2"
    )

    ax.set_title(
        "K-Medoids Clustering using PCA"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)


    # --------------------------------------------------------
    # MEDOIDS
    # --------------------------------------------------------

    st.subheader(
        "K-Medoids"
    )

    medoids_original = X.iloc[
        medoid_indices
    ]

    medoids_original = medoids_original.copy()

    medoids_original.index = [
        "Medoid " + str(i)
        for i in range(k)
    ]

    st.dataframe(
        medoids_original,
        use_container_width=True
    )


    st.write(
        "Medoid indices:",
        medoid_indices
    )


    # --------------------------------------------------------
    # CLUSTER SIZE
    # --------------------------------------------------------

    st.subheader(
        "Cluster Distribution"
    )

    cluster_counts = pd.Series(
        labels
    ).value_counts().sort_index()

    cluster_counts.index = [
        "Cluster " + str(i)
        for i in cluster_counts.index
    ]

    st.bar_chart(
        cluster_counts
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "K-Medoids Interpretation"
    )

    st.write(
        "K-Medoids selects actual observations from the dataset "
        "as cluster representatives. Therefore, medoids are "
        "real data points rather than calculated averages."
    )

    st.info(
        "K-Medoids is generally more robust to outliers "
        "than K-Means."
    )


# ============================================================
# COMPARISON SECTION
# ============================================================

else:

    st.header(
        "7. K-Means vs K-Medoids Comparison"
    )

    # Run K-Means

    (
        kmeans,
        kmeans_labels,
        centroids,
        kmeans_silhouette,
        kmeans_db,
        kmeans_ch
    ) = run_kmeans(k)


    # Run K-Medoids

    (
        kmedoids,
        kmedoids_labels,
        medoid_indices,
        medoids,
        kmedoids_silhouette,
        kmedoids_db,
        kmedoids_ch
    ) = run_kmedoids(k)


    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    comparison = pd.DataFrame({

        "Metric": [
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index"
        ],

        "K-Means": [
            kmeans_silhouette,
            kmeans_db,
            kmeans_ch
        ],

        "K-Medoids": [
            kmedoids_silhouette,
            kmedoids_db,
            kmedoids_ch
        ]
    })


    st.subheader(
        "Evaluation Metric Comparison"
    )

    st.dataframe(
        comparison.round(4),
        use_container_width=True
    )


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    st.subheader(
        "Performance Summary"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🔵 K-Means"
        )

        st.metric(
            "Silhouette",
            round(
                kmeans_silhouette,
                4
            )
        )

        st.metric(
            "Davies-Bouldin",
            round(
                kmeans_db,
                4
            )
        )

        st.metric(
            "Calinski-Harabasz",
            round(
                kmeans_ch,
                4
            )
        )


    with col2:

        st.markdown(
            "### 🟣 K-Medoids"
        )

        st.metric(
            "Silhouette",
            round(
                kmedoids_silhouette,
                4
            )
        )

        st.metric(
            "Davies-Bouldin",
            round(
                kmedoids_db,
                4
            )
        )

        st.metric(
            "Calinski-Harabasz",
            round(
                kmedoids_ch,
                4
            )
        )


    # --------------------------------------------------------
    # PCA COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "Cluster Visualization Comparison"
    )

    centroids_pca = pca.transform(
        centroids
    )

    medoids_pca = pca.transform(
        medoids
    )


    col1, col2 = st.columns(2)


    # K-Means plot

    with col1:

        st.write(
            "### K-Means"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=kmeans_labels,
            cmap="viridis",
            s=50
        )

        ax.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            marker="X",
            s=200,
            c="red",
            label="Centroids"
        )

        ax.set_xlabel(
            "PC1"
        )

        ax.set_ylabel(
            "PC2"
        )

        ax.set_title(
            "K-Means"
        )

        ax.legend()

        ax.grid(True)

        st.pyplot(fig)


    # K-Medoids plot

    with col2:

        st.write(
            "### K-Medoids"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=kmedoids_labels,
            cmap="plasma",
            s=50
        )

        ax.scatter(
            medoids_pca[:, 0],
            medoids_pca[:, 1],
            marker="X",
            s=200,
            c="red",
            label="Medoids"
        )

        ax.set_xlabel(
            "PC1"
        )

        ax.set_ylabel(
            "PC2"
        )

        ax.set_title(
            "K-Medoids"
        )

        ax.legend()

        ax.grid(True)

        st.pyplot(fig)


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.subheader(
        "Comparison Interpretation"
    )

    st.write(
        """
        **Silhouette Score:** Higher values indicate better
        defined and separated clusters.

        **Davies-Bouldin Index:** Lower values indicate better
        clustering.

        **Calinski-Harabasz Index:** Higher values indicate
        better separated and compact clusters.

        **K-Means:** Uses calculated centroids based on the mean
        of observations.

        **K-Medoids:** Uses actual observations as cluster
        representatives and is generally more resistant to
        outliers.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MAI511-2 Advanced Machine Learning | "
    "Lab Exercise 7 | K-Means and K-Medoids"
)