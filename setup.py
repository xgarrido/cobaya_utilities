from setuptools import find_packages, setup

import versioneer

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="cobaya_utilities",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    description="A set of functions to deal with MCMC output from cobaya",
    long_description=readme,
    long_description_content_type="text/x-rst",
    url="https://github.com/xgarrido/cobaya_utilities.git",
    author="Xavier Garrido",
    author_email="xavier.garrido@lal.in2p3.fr",
    keywords=["cobaya", "MCMC", "plot"],
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Education",
    ],
    install_requires=["cobaya", "seaborn"],
)
