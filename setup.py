from setuptools import setup, find_packages

setup(
    name='SMOPCA',
    version='0.1.0',
    author='Mo Chen',
    author_email='mochen@smail.nju.edu.cn',
    packages=find_packages(),
    description='A novel spatially aware multi-omics dimension reduction method',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/cmhimself/SMOPCA',
    install_requires=[
        'anndata==0.8.0',
        'h5py==3.8.0',
        'matplotlib==3.6.3',
        'numpy==1.23.5',
        'pandas==1.5.3',
        'scanpy==1.9.1',
        'scikit-learn==1.2.1',
        'scipy==1.10.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10.0',
)
