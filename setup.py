from setuptools import setup, find_packages

setup(
    name='SpaMosaic',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='MIT',
    author='Jin Zuoyou',
    author_email='0921160212@csu.edu.cn',
    description='A computational framework for spatial mosaic integration.',
    package_data={
        'spamosaic': ['configs/*.yaml']
    },
)