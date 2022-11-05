from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='cnvm',
      version='0.1.0',
      description='Continuous-time noisy voter model',
      long_description=readme,
      author='Marvin Lücke',
      author_email='luecke@zib.de',
      license=license,
      packages=find_packages()
      )
