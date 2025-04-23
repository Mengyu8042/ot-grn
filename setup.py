# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# Read the README.md file
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name='ot-grn',
    version='1.0.1',
    description='Double Optimal Transport for Differential Gene Regulatory Network Inference with Unpaired Samples',
    long_description=long_description,  # Add the README content
    long_description_content_type="text/markdown",  # Specify the content type (Markdown)
    author='Mengyu Li',
    author_email='limengyu516@ruc.edu.cn',
    url='https://github.com/Mengyu8042/ot-grn',
    packages=find_packages(include=["ot_grn", "ot_grn.data"]), 
    include_package_data=True,
    package_data={
        "ot_grn": ["data/*.csv"], 
    },
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'pot',  # Python Optimal Transport library
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'black',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.7', 
)
