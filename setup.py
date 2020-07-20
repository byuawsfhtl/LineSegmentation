from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='lineseg',
    version='0.1',
    description='Line-Level Segmentation using an ARU-Net architecture',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BYU-Handwriting-Lab/LineSegmentation',
    author='BYU-Handwriting-Lab',
    keywords='BYU segmentation line handwriting recognition',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['tensorflow', 'pillow', 'scikit-learn', 'tqdm', 'matplotlib', 'shapely', 'numpy']
)
