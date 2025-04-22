import setuptools

REPO_NAME = "Mylib"  # tên của github repo
SRC_REPO = "Mylib"
AUTHOR_USER_NAME = "zalexdevon"  # tên của tài khoản github
AUTHOR_EMAIL = "trantamch112358@gmail.com"  # email đăng kí github

__version__ = "0.0.0"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
