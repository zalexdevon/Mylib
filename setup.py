import setuptools

SRC_REPO = "Mylib"  # tên của github repo
AUTHOR_USER_NAME = "zalexdevon"  # tên của tài khoản github
AUTHOR_EMAIL = "trantamch112358@gmail.com"  # email đăng kí github

__version__ = "0.0.0"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{SRC_REPO}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
