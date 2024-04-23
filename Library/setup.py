from setuptools import setup, find_packages
    
setup(
    name = 'Memory Mosaics',
    version = "0.1",
    author = ['Jianyu Zhang','Leon Bottou'], 
    author_email = ['jianyu@nyu.edu','leon@bottou.org'],
    #url = '',
    description = 'Memory Mosaics for PyTorch',
    license = "MIT license",
    packages = find_packages(exclude=["test"]),  # Don't include test directory in binary distribution
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
    ]  # Update these accordingly
)