from setuptools import setup

setup(
    name="gym_maze",
    version="1.0.0",
    author="Matthias Konitzny",
    description="Adds additional environments to the OpenAI Gym package",
    install_requires=["gym", "numpy"],
    include_package_data=True,
    package_data={"": ["mapdata/*.csv"]},
)
