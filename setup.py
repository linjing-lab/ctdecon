import os

from setuptools import setup

try:
    pkg_name = 'ctdecon'
    libinfo_py = os.path.join(pkg_name, '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf-8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
    exec(version_line) # gives __version
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
      name='ctdecon',  # pkg_name
      packages=['ctdecon',],
      version=__version__,  # version number
      description="Reference-based cell type deconvolution in spatial transcriptomics.",
      author='林景',
      author_email='linjing010729@163.com',
      license='MIT',
      url='https://github.com/linjing-lab/ctdecon',
      download_url='https://github.com/linjing-lab/ctdecon/tags',
      long_description=_long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      zip_safe=False,
      setup_requires=['setuptools>=18.0', 'wheel'],
      project_urls={
            'Source': 'https://github.com/linjing-lab/ctdecon/tree/master/ctdecon/',
            'Tracker': 'https://github.com/linjing-lab/ctdecon/issues',
      },
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=[
            'numpy>=1.26.4',
            'pandas>=2.3.1',
            'POT>=0.9.5',
            'pykan>=0.2.8',
            'scanpy>=1.10.4',
            'scikit-learn>=1.7.1',
            'scipy>=1.11.4',
            'tqdm>=4.67.1'
      ],
      # extras_require=[]
)