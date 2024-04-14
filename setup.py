from setuptools import setup, find_packages

setup(
    name='temperature_arima',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'statsmodels'
    ],
    author='Ihr Name',
    description='Eine Zeitreihenanalyse der Temperaturdaten mit ARIMA',
    long_description=open('README.md').read(),
    url='https://github.com/IhrGithub/temperature_arima',
)
