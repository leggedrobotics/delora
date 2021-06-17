import os
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstallCommand(install):
    # This is only run for "python setup.py install" (not for "pip install -e .")
    def run(self):
        print("--------------------------------")
        print("Writing environment variables to .env ...")
        os.system('echo "export DELORA_ROOT=' + os.getcwd() + '" >> .env')
        print("...done.")
        print("--------------------------------")
        install.run(self)


setup(name='DeLORA',
      version='1.0',
      author='Julian Nubert (nubertj@ethz.ch)',
      package_dir={"": "src"},
      install_requires=[
          'numpy',
          'torch',
          'opencv-python',
          'pyyaml',
          'rospkg'
      ],
      scripts=['bin/preprocess_data.py', 'bin/run_rosnode.py', 'bin/run_testing.py', 'bin/run_training.py',
               'bin/visualize_pointcloud_normals.py'],
      license='LICENSE',
      description='Self-supervised Learning of LiDAR Odometry for Robotic Applications',
      cmdclass={'install': CustomInstallCommand, },
      )
