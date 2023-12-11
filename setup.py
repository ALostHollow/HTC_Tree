from setuptools import setup, find_packages

setup(
    name="htc_tree",
    version="1.0.0",
    description="A structure to represent hierarchical text classification problems",
    packages=find_packages(include=['src/','Docs/', 'README.md']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
