from setuptools import setup, find_packages

setup(
    name="Mylib",
    version="0.1",
    packages=find_packages(where="Mylib"),
    install_requires=[
        "tensorflow==2.18.0",
        "keras-cv==0.9.0",
        "pandas",
        "numpy",
        "matplotlib",
        "mlflow==2.2.2",
        "scikit-learn==1.3.0",  # phiên bản này phù hợp với xgboost
        "python-box==6.0.2",
        "pyYAML",
        "ensure==1.0.2",
        "types-PyYAML",
        "plotly",
        "seaborn",
        "xgboost",
        "lightgbm",
        "imbalanced-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
