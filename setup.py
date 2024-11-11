from setuptools import find_packages, setup

setup(
    name='Dynamic Price Predicition',
    version='0.0.1',
    author='Nithin',
    author_email='cherukumallinithin@gmail.com',
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy"
    ],
    packages=find_packages()
)