from setuptools import setup, find_packages

setup(
    name="youtube-search-api",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    extras_require={
        "dev": [
            "pytest",
        ]
    },
    author="Filip Ccederquist",
    author_email="cederquist94@hotmail.com",
    description="A way to search transcripts from youtube videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cerre/youtube-search-api/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
