import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dynamic_indicators_tools",
    version="1.0.2",
    author="Santiago Arranz Sanz",
    description="Repositorio que recoge herramientas para el análisis de sistemas dinámicos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src"),
    url="https://github.com/santhiperbolico/dynamic_indicators_tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    py_modules=["dynamic_indicators_tools"],
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "matplotlib", "attrs", "tqdm", "pyqt5"],
)
