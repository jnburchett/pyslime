from setuptools import setup, find_packages
import os
import glob
import codecs
import os.path


setup_keywords = dict()
setup_keywords["name"] = "pyslime"


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup_keywords["version"] = get_version("pyslime/__init__.py")

setup_keywords[
    "description"
] = "Tools for analyzing output of Monte Carlo Physarum Machine models"

# setup_keywords["packages"] = find_packages("pyslime")
setup_keywords["packages"] = find_packages()

setup_keywords["long_description"] = ""
if os.path.exists("README.md"):
    with open("README.md") as readme:
        setup_keywords["long_description"] = readme.read()


# find the scripts
if os.path.isdir("bin"):
    setup_keywords["scripts"] = [fname for fname in glob.glob(os.path.join("bin", "*"))]

data_files = []
# walk through the data directory, adding all files
data_generator = os.walk("pyslime/data")
for path, directories, files in data_generator:
    for f in files:
        data_path = "/".join(path.split("/")[1:])
        data_files.append(data_path + "/" + f)
setup_keywords["package_data"] = {
    "pyslime": data_files,
    "": ["*.rst", "*.txt", "*.yaml"],
}
setup_keywords["include_package_data"] = True

setup(**setup_keywords)

