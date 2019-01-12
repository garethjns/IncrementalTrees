import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(name='incremental_trees',
                 version="0.0.1",
                 author="Gareth Jones",
                 author_email="author@example.com",
                 description='Sklearn forests with partial fits',
                 # long_description=long_description,
                 # long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 url="",
                 install_requires=['dask', 'dask_ml', 'sklearn', 'bokeh'])
