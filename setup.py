from setuptools import setup

setup(
    name='Py-Audio-Recording',
    version='0.0.0',
    packages=['arlib'],
    install_requires=[
        'numpy~=1.24',
        'aiohttp~=3.8',
        'scipy~=1.10',
        'sounddevice~=0.4',
        'pymongo[srv]~=4.4',
        'librosa~=0.10'
    ],
    url="https://github.com/deadbeef-development/Py-Audio-Recording",
    author='deadbeef-development',
    author_email='deadbeef.development@gmail.com'
)
