import setuptools

with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]
    print(INSTALL_REQUIRES)

def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(
    name="silkscreen",
    version="0.1",
    author="Tim Miller, Imad Pasha, Ava Polzin",
    author_email="tim.miller@yale.edu",
    description="SilkScreen:Using sbi and artpop to measure properties of dwarf galaxies",
    long_description=readme(),
    long_description_content_type="text/x-rst",
    url="https://github.com/tbmiller-astro/SilkScreen",
    entry_points = {},
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"],
    license='MIT',

)
