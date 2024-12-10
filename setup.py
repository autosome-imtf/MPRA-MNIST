from setuptools import setup

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

setup(name='mpramnist',
      version="0.1.2",
      description='available MPRA datasets',
      url='https://github.com/autosome-imtf/MpraDataset',
      author="Nikita P",
      author_email='',
      long_description=readme(),
      long_description_content_type="text/markdown",
      license='',
      packages = ['mpramnist'],
      zip_safe=True)