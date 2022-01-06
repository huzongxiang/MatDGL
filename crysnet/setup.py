from setuptools import setup
from setuptools import find_packages


setup(
    name='crysnet',
    version='0.0.1',
    description='Labelled Graph Networks for machine learning of crystal.',
    long_description='using mpnn or graphtransformer to learn',
    long_description_content_type='text/markdown',
    author='Zongxiang Hu',
    author_email='huzongxiang@yahoo.com',
    download_url='https://github.com/materialsvirtuallab/megnet',
    license='BSD',
    install_requires=['numpy', "scikit-learn",
                      'pymatgen>=2019.10.4'],
    extras_require={
        'tensorflow': ['tensorflow=2.6'],
        'tensorflow with gpu': ['tensorflow-gpu=2.6'],
    },
    packages=find_packages(),
    package_data={
        "crysnet": ["*.json", "*.md"],
    },
    include_package_data=True,
    keywords=["materials", "science", "machine", "learning", "deep", "graph", "networks", "neural", "transformer", "massagepassing"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
