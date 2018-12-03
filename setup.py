from setuptools import setup

# Should match git tag
VERSION = '0.0.1'

def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as file:
    REQUIRED_MODULES = [line.strip() for line in file]

with open('requirements-dev.txt') as file:
    DEVELOPMENT_MODULES = [line.strip() for line in file]


setup(name='data_preprocessing_arch',
      version=VERSION,
      description='Python module for preprocessing data',
      long_description=readme(),
      keywords='markdown to html',
      url='https://github.com/mecheverria96/archMLP-Preprocessing',
      author='Mariana Echeverria',
      author_email='mecheverria@middlebury.edu',
      packages=['arch_dp'],
      install_requires=REQUIRED_MODULES,
      extras_require={'dev': DEVELOPMENT_MODULES},
      include_package_data=True)