from setuptools import setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


setup(
    name="mpramnist",
    version="1.0",
    description="standardized collection of MPRA datasets",
    url="https://github.com/autosome-imtf/MPRA-MNIST",
    author=["Nikita Penzin", "Arsenii ZinkevichIvan Kulakovckiy", "Dmitry Penzar"],
    author_email=["nios.583@gmail.com", "-", "-", "dmitrypenzar1996@gmail.com"],
    long_description=readme(),
    long_description_content_type="text/markdown",
    license="",
    packages=["mpramnist"],
    zip_safe=True,
)
