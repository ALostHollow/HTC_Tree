from setuptools import setup, find_packages

setup(
    name="htc_tree",
    version="0.0.3",
    author="Rowan Andruko",
    description="A framework for Hierarchical Text Classification.",
    long_description="A framework for Hierarchical Text Classification. I WILL ADD A LONG DESC LATER",
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
      ],
    python_requires=">=3.11",
    keywords=['python', 'text classification', 'hierarchical', 'text labeling']
)
