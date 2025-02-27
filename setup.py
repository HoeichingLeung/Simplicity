import os
import setuptools  # 没有的直接pip install一下就行了

setuptools.setup(
    name='MySDK',
    version='1.0',
    description='A MySDK for python packaging.',  # 一个简要的介绍而已
    long_description=open(
        os.path.join(
            os.path.dirname(__file__),
            'README.rst'
        )
    ).read(),
    packages=setuptools.find_packages(),
    include_package_data=True,
    author='xxx',
    author_email='xxx@gmail.com',
)
