from setuptools import setup, find_packages

VERSION = '0.1.1' 
DESCRIPTION = 'Python InfluxDB Class'
LONG_DESCRIPTION = 'Python3 package to serve as the interface to InfluxDB for Multiple Sclerosis data'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="InfluxDBms",
        version=VERSION,
        author="Angela Fernandez",
        author_email="j.ordieres@upm.es",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'

        keywords=['python', 'InfluxDB access package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
