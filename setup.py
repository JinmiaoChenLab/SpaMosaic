from setuptools import setup, find_packages

setup(
    name='SpaMosaic',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='MIT',
    author='Jinmiao Lab',
    author_email='204707014@csu.edu.cn',
    description='Mosaic integration of spatial multi-omics with SpaMosaic.',
    package_data={
        'spamosaic': ['configs/*.yaml']
    },
)