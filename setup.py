from setuptools import find_packages, setup

setup(
    name="cobaya_utilities",
    version="0.1",
    packages=find_packages(),
    description="A set of functions to deal with MCMC output from cobaya",
    url="https://github.com/xgarrido/cobaya_utilities.git",
    author="Xavier Garrido",
    author_email="xavier.garrido@lal.in2p3.fr",
    keywords=["cobaya", "MCMC", "plot"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Education",
    ],
    install_requires=["cobaya"],
)
