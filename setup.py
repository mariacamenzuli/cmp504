from setuptools import setup

setup(name='cmp504',
      version='0.1',
      description='A visual GUI testing library.',
      url='https://github.com/maria-camenzuli/cmp504',
      author='Maria Camenzuli',
      license='MIT',
      packages=['cmp504'],
      install_requires=[
          'opencv-contrib-python',
          'numpy',
          'mss',
          'Pillow',
          'matplotlib'
      ],
      test_suite='nose2.collector.collector',
      tests_require=[
          'nose2',
          'assertpy'
      ],
      include_package_data=True,
      zip_safe=False)
