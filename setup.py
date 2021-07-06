from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'SIR compartment modelling framework'
LONG_DESCRIPTION = 'Framework for creating stochastic SIR-style compartment models and running Monte Carlo simulations.'

# Setting up
setup(
        name="sirplus", 
        version=VERSION,
        author="Stephen Okazawa",
        author_email="okazawa.s@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'matplotlib', 'pandas'],
        
        keywords=['python', 'sir', 'model', 'simulation'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)
