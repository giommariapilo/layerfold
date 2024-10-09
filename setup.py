import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

__version__ = "1.0"

setup(
    name='torch-layerfold',
    version=__version__,
    description='Depth compression of DNNs for accelerated inference',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/giommariapilo/layerfold',
    author='Giommaria Pilo',
    author_email='giommaria.pilo@telecom-paris.fr',
    license='MIT License',
    packages=find_packages(exclude=('.github', 'tests')),
    zip_safe=False,
    install_requires=[
        'torch',
        'torchvision',
    ],
    python_requires='>=3.10'
)