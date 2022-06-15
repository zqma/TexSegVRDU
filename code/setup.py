import setuptools
with open("../README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setuptools.setup(
    name="TEXSEGVRDU",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Zhiqiang",
    author_email="zhiqiang.a.ma@gmail.com",
    install_requires=[],
    version="0.0.1",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",  

    ],
    packages=setuptools.find_namespace_packages(),
    package_data={"":["*.ini"]},
    include_package_data=True,
    python_requires=">=3.8"
)

