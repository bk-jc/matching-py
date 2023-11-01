from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="jarvis2",
    version="0.0.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Bas Krahmer",
    author_email="bas.krahmer@jobcloud.ch",
    description="Training, data and inference code for Jarvis V2",
)
