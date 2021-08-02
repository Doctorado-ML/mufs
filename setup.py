import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


def get_data(field: str):
    item = ""
    with open("mufs/__init__.py") as f:
        for line in f.readlines():
            if line.startswith(f"__{field}__"):
                delim = '"' if '"' in line else "'"
                item = line.split(delim)[1]
                break
        else:
            raise RuntimeError(f"Unable to find {field} string.")
    return item


setuptools.setup(
    name="MUFS",
    version=get_data("version"),
    license=get_data("license"),
    description="Multi Feature Selection",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/Doctorado-ML/mufs#mufs",
    project_urls={
        "Code": "https://github.com/Doctorado-ML/mufs",
    },
    author=get_data("author"),
    author_email=get_data("author_email"),
    keywords="feature-selection",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: " + get_data("license"),
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["scikit-learn"],
    test_suite="mufs.tests",
    zip_safe=False,
)
