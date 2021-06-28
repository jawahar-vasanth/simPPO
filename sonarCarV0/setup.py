import setuptools

# define the name to import the env
# import gym_racer
setuptools.setup(
    name="sonarCarV0",
    version="0.0.0",
    author="Jawahar",
    author_email="jawahar.vasanth@gmail.com",
    description="OpenAI safety gym environment of a racing car.",
    packages=setuptools.find_packages(where="."),
    install_requires=["gym", "pygame", "numpy", "pymunk"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
