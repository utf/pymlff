from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_desc = f.read()

setup(
    name="pymlff",
    description="Python package for reading and writing VASP ML_AB files.",
    long_description=long_desc,
    use_scm_version={"version_scheme": "python-simplified-semver"},
    setup_requires=["setuptools_scm"],
    long_description_content_type="text/markdown",
    author="Alex Ganose",
    author_email="alexganose@gmail.com",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"pymlff": ["py.typed"]},
    zip_safe=False,
    include_package_data=True,
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Other/Nonlisted Topic",
        "Topic :: Database :: Front-Ends",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    tests_require=["pytest"],
)
