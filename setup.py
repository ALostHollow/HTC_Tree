from setuptools import setup, find_packages

setup(
    name="htc_tree",
    version="0.0.1",
    author="Rowan Andruko",
    description="A framework for Hierarchical Text Classification.",
    packages=find_packages(exclude=['tests', 'tests.*', '.git*', 'docs']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'pandas',
          'scikit-learn',
          'xgboost',
          'matplotlib',
          'pickle'

      ],
    python_requires=">=3.11",
    keywords=['python', 'text classification', 'hierarchical', 'text labeling']
)
