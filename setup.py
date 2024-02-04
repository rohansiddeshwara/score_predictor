from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    with open('requirements.txt') as f:
        return f.read().splitlines()
    

setup(
name='score_predictor',
version='0.0.1',
author='rohan',
author_email='rohansiddeshwara@gmail.com',
packages=find_packages(),
install_requires=get_requirements().remove("-e .")
)