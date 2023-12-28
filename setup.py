'''
This script is used to build the entire ML application as a package.
It can also be used to deploy the package to PyPI.
'''

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'   # This is present in requirements.txt to connect it to setup.py

def get_requirements(req_filepath: str) -> List[str]:
    '''
    Function to return a list of required packages.
    '''
    requirements = []
    with open(req_filepath, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)   # Remove the "-e ." while running requirements.txt from setup.py
                                                # The "-e ." is just required to tell setup.py to install requirements.txt
            
    return requirements

setup(
    name='student_performance',
    version='0.0.1',
    author='Vipul',
    author_email='vipulupadhye@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)