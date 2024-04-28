#!/usr/bin/env python

import setuptools
from distutils.core import setup

dependencies = [
    "numpy"
]

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name = "phylokrr",
      version = '0.4.2',
      long_description = readme,
      long_description_content_type = 'text/markdown',
      packages = ['phylokrr'],
      package_dir = {'phylokrr': 'src'},
      python_requires='>=3.5',
      install_requires = dependencies,
      zip_safe = False,
      classifiers = [
          'Programming Language :: Python :: 3',
      ]
    )


