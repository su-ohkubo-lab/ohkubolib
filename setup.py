from setuptools import setup

def version(path: str) -> str:
    with open(path, 'r') as f:
        _version: str = f.read().strip()
    return _version

def install_requires(path: str) -> list[str]:
    requirements: list[str] = list[str]()
    with open(path, 'r') as f:
        for requirement in f:
            requirements.append(requirement.strip())
    return requirements

setup(
    name='ohkubolib',
    version=version('version'),
    install_requires=install_requires('requirements.txt'),
)