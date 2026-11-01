# Linear Algebra-Based Data Analytics Framework

**A Python-based interactive data analytics application for dimensionality reduction and pattern discovery using Linear Algebra, Principal Component Analysis (PCA), and K-Means Clustering.**

## About the Project

This project focuses on applying mathematical concepts from Linear Algebra to practical data analytics.

The framework allows users to upload their own CSV datasets, explore their data, perform preprocessing, reduce dimensionality using Principal Component Analysis, and discover hidden patterns through K-Means clustering.

The application is developed using Python and Streamlit, providing an interactive interface for performing different analytical operations.

## Key Features

- **Dataset Upload:** Upload custom CSV datasets for analysis.
- **Data Exploration:** View dataset structure, statistical summaries, and basic visualizations.
- **Data Preprocessing:** Prepare numerical data and standardize features.
- **Linear Algebra Operations:** Calculate covariance matrices, eigenvalues, and eigenvectors.
- **Principal Component Analysis (PCA):** Reduce dataset dimensionality while retaining important information.
- **K-Means Clustering:** Group similar data points to identify patterns.
- **Data Visualization:** Present analytical results through visual representations.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation |
| NumPy | Numerical and linear algebra operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning algorithms |

## Project Structure

```text
linear-algebra-data-analytics-framework/
│
├── app.py
├── requirements.txt
├── sample_dataset.csv
│
└── modules/
    ├── clustering.py
    ├── linear_algebra.py
    ├── pca.py
    └── preprocessing.py
```

## Mathematical Concepts Implemented

**1. Covariance Matrix**

Used to understand relationships between numerical variables in the dataset.

**2. Eigenvalues and Eigenvectors**

Used in understanding the directions of maximum variance and the mathematical foundation of PCA.

**3. Principal Component Analysis (PCA)**

Transforms high-dimensional numerical data into fewer dimensions while preserving as much relevant variance as possible.

**4. K-Means Clustering**

An unsupervised learning algorithm that groups data points based on similarity.

## Installation and Execution

1. Clone the repository:

```bash
git clone https://github.com/Arhaan-k20/linear-algebra-data-analytics-framework.git
```

2. Navigate to the project directory:

```bash
cd linear-algebra-data-analytics-framework
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python -m streamlit run app.py
```

5. Open the local Streamlit URL displayed in your terminal.

## Project Objective

The primary objective is to demonstrate how Linear Algebra concepts can be integrated into a practical data analytics workflow, connecting mathematical foundations with dimensionality reduction, machine learning, and pattern discovery.

## Author

**Arhaan Khan**

BSc Data Science and Big Data Analytics

GitHub: [Arhaan-k20](https://github.com/Arhaan-k20)

---

*Developed as an academic final-year project.*
