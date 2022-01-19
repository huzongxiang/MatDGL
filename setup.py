from setuptools import setup
from setuptools import find_packages

import os

this_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='crysnet',
    version='0.1.7',
    description='Labelled Graph Networks for machine learning of crystal.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Zongxiang Hu',
    author_email='huzongxiang@yahoo.com',
    download_url='https://github.com/huzongxiang/CrysNetwork',
    license='BSD',
    install_requires=['numpy', "scikit-learn"],
    packages=find_packages(),
    package_data={
        "crysnet": ["data/*.json", "models/model/*.hdf5", "*.png"],
    },
    include_package_data=True,
    keywords=["materials", "science", "machine", "learning", "deep", "graph", "networks", "neural", "transformer", "massagepassing", "topology", "tight", "bingding", "twisted", "graphene"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)