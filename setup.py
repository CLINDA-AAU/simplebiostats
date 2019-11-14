import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplebiostats",  # Replace with your own username
    version="0.0.1",
    author="Charles Vesteghem",
    author_email="charles.vesteghem@rn.dk",
    description="Simple Biostatistics in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HaemAalborg/simplebiostats",
    packages=setuptools.find_packages(),
    licence='BSD License',
    setup_requires=['statsmodels>=0.10.1'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
