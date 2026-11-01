import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modules.preprocessing import (
    prepare_numeric_data,
    standardize_data
)

from modules.linear_algebra import (
    calculate_covariance_matrix,
    calculate_eigenvalues_eigenvectors
)

from modules.pca import (
    calculate_pca,
    transform_data
)

from modules.clustering import (
    perform_kmeans,
    calculate_silhouette_score
)


st.set_page_config(
    page_title="Linear Algebra-Based Data Analytics Framework",
    page_icon="📊",
    layout="wide"
)

st.title("Linear Algebra-Based Data Analytics Framework")

st.write(
    "A data analytics framework using Linear Algebra, "
    "Principal Component Analysis, and K-Means clustering."
)

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()

if df.empty:
    st.warning("The uploaded dataset is empty.")
    st.stop()


# ==========================================================
# PHASE 1 — DATASET ANALYSIS
# ==========================================================

st.header("Phase 1 — Dataset Analysis")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")

with col2:
    st.metric("Columns", f"{df.shape[1]:,}")

with col3:
    st.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )

with col4:
    st.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )


st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


st.subheader("Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Non-Null Values": df.notna().sum().values,
    "Missing Values": df.isna().sum().values,
    "Unique Values": df.nunique(dropna=True).values
})

st.dataframe(
    column_info,
    use_container_width=True
)


numeric_df = df.select_dtypes(
    include=np.number
)


st.subheader("Descriptive Statistics")

if numeric_df.empty:

    st.warning(
        "No numerical columns were detected."
    )

else:

    statistics = pd.DataFrame({
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Mode": numeric_df.mode().iloc[0],
        "Variance": numeric_df.var(),
        "Standard Deviation": numeric_df.std(),
        "Minimum": numeric_df.min(),
        "Q1": numeric_df.quantile(0.25),
        "Q2 (Median)": numeric_df.quantile(0.50),
        "Q3": numeric_df.quantile(0.75),
        "Maximum": numeric_df.max()
    })

    st.dataframe(
        statistics.round(4),
        use_container_width=True
    )


st.subheader("Missing Value Analysis")

missing_values = df.isna().sum()

missing_values = missing_values[
    missing_values > 0
].sort_values(
    ascending=False
)

if missing_values.empty:

    st.success(
        "No missing values were found."
    )

else:

    missing_table = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values,
        "Missing Percentage": (
            missing_values.values /
            len(df) * 100
        ).round(2)
    })

    st.dataframe(
        missing_table,
        use_container_width=True
    )


st.subheader("Feature Distribution")

if not numeric_df.empty:

    selected_feature = st.selectbox(
        "Select a numerical feature",
        numeric_df.columns
    )

    values = numeric_df[
        selected_feature
    ].dropna()

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.hist(
        values,
        bins=20,
        edgecolor="black"
    )

    ax.set_title(
        f"Distribution of {selected_feature}"
    )

    ax.set_xlabel(
        selected_feature
    )

    ax.set_ylabel(
        "Frequency"
    )

    fig.tight_layout()

    st.pyplot(fig)


st.subheader("Correlation Analysis")

if numeric_df.shape[1] >= 2:

    correlation_matrix = numeric_df.corr()

    st.dataframe(
        correlation_matrix.round(3),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        x_feature = st.selectbox(
            "Select X-axis feature",
            numeric_df.columns,
            key="x_feature"
        )

    with col2:

        y_feature = st.selectbox(
            "Select Y-axis feature",
            numeric_df.columns,
            index=1,
            key="y_feature"
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.scatter(
        numeric_df[x_feature],
        numeric_df[y_feature],
        alpha=0.65
    )

    ax.set_title(
        f"{x_feature} vs {y_feature}"
    )

    ax.set_xlabel(
        x_feature
    )

    ax.set_ylabel(
        y_feature
    )

    fig.tight_layout()

    st.pyplot(fig)


# ==========================================================
# PHASE 2 — LINEAR ALGEBRA
# ==========================================================

st.divider()

st.header("Phase 2 — Linear Algebra Analysis")

try:

    prepared_data = prepare_numeric_data(
        df
    )

    standardized_data = standardize_data(
        prepared_data
    )

    st.subheader("Numerical Feature Matrix")

    st.write(
        "The numerical variables are extracted "
        "and represented as a feature matrix."
    )

    st.dataframe(
        prepared_data.head(10),
        use_container_width=True
    )


    st.subheader("Standardized Feature Matrix")

    st.write(
        "Standardization places the numerical "
        "features on a comparable scale."
    )

    st.dataframe(
        standardized_data.head(10).round(4),
        use_container_width=True
    )


    st.subheader("Covariance Matrix")

    covariance_matrix = calculate_covariance_matrix(
        standardized_data
    )

    covariance_df = pd.DataFrame(
        covariance_matrix,
        index=prepared_data.columns,
        columns=prepared_data.columns
    )

    st.dataframe(
        covariance_df.round(4),
        use_container_width=True
    )


    st.subheader(
        "Eigenvalues and Eigenvectors"
    )

    eigenvalues, eigenvectors = (
        calculate_eigenvalues_eigenvectors(
            covariance_matrix
        )
    )

    eigenvalue_df = pd.DataFrame({
        "Principal Component": [
            f"PC{i + 1}"
            for i in range(len(eigenvalues))
        ],
        "Eigenvalue": eigenvalues
    })

    st.dataframe(
        eigenvalue_df.round(4),
        use_container_width=True
    )


    eigenvector_df = pd.DataFrame(
        eigenvectors,
        index=prepared_data.columns,
        columns=[
            f"PC{i + 1}"
            for i in range(
                eigenvectors.shape[1]
            )
        ]
    )

    st.write("Eigenvectors:")

    st.dataframe(
        eigenvector_df.round(4),
        use_container_width=True
    )


    # ======================================================
    # PHASE 3 — PCA
    # ======================================================

    st.divider()

    st.header(
        "Phase 3 — Principal Component Analysis"
    )

    (
        pca_covariance,
        pca_eigenvalues,
        pca_eigenvectors,
        explained_variance_ratio,
        cumulative_variance
    ) = calculate_pca(
        standardized_data
    )


    pca_table = pd.DataFrame({
        "Component": [
            f"PC{i + 1}"
            for i in range(
                len(pca_eigenvalues)
            )
        ],
        "Eigenvalue": pca_eigenvalues,
        "Explained Variance (%)":
            explained_variance_ratio * 100,
        "Cumulative Variance (%)":
            cumulative_variance * 100
    })


    st.subheader(
        "Explained Variance"
    )

    st.dataframe(
        pca_table.round(4),
        use_container_width=True
    )


    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        range(
            1,
            len(cumulative_variance) + 1
        ),
        cumulative_variance * 100,
        marker="o"
    )

    ax.set_xlabel(
        "Number of Principal Components"
    )

    ax.set_ylabel(
        "Cumulative Explained Variance (%)"
    )

    ax.set_title(
        "Cumulative Explained Variance"
    )

    ax.grid(True)

    fig.tight_layout()

    st.pyplot(fig)


    # ======================================================
    # PCA TRANSFORMATION
    # ======================================================

    st.subheader(
        "Reduced-Dimensional Dataset"
    )

    n_components = st.slider(
        "Select number of principal components",
        min_value=2,
        max_value=min(
            5,
            standardized_data.shape[1]
        ),
        value=2
    )


    transformed_data = transform_data(
        standardized_data,
        pca_eigenvectors,
        n_components
    )


    reduced_df = pd.DataFrame(
        transformed_data,
        columns=[
            f"PC{i + 1}"
            for i in range(n_components)
        ]
    )


    st.dataframe(
        reduced_df.head(10).round(4),
        use_container_width=True
    )


    # ======================================================
    # PHASE 4 — K-MEANS CLUSTERING
    # ======================================================

    st.divider()

    st.header(
        "Phase 4 — Pattern Discovery Using K-Means"
    )


    n_clusters = st.slider(
        "Select number of clusters",
        min_value=2,
        max_value=5,
        value=3
    )


    clustering_model, labels = perform_kmeans(
        transformed_data,
        n_clusters
    )


    clustered_df = reduced_df.copy()

    clustered_df["Cluster"] = labels


    st.subheader(
        "Clustered Dataset"
    )

    st.dataframe(
        clustered_df.head(10),
        use_container_width=True
    )


    silhouette = calculate_silhouette_score(
        transformed_data,
        labels
    )


    if silhouette is not None:

        st.metric(
            "Silhouette Score",
            f"{silhouette:.4f}"
        )


    if n_components >= 2:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        scatter = ax.scatter(
            transformed_data[:, 0],
            transformed_data[:, 1],
            c=labels,
            alpha=0.75
        )

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

        ax.set_title(
            "K-Means Clusters in PCA Space"
        )

        fig.tight_layout()

        st.pyplot(fig)


except Exception as error:

    st.error(
        f"Linear Algebra analysis could not be completed: {error}"
    )


st.divider()

st.success(
    "Data analysis and Linear Algebra processing completed successfully."
)