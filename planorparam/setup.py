#!/usr/bin/env python
  
from distutils.core import setup
#from catkin_pkg.python_setup import generate_distutils_setup

#setup_args = generate_distutils_setup
setup(name='planorparam',
      version='2.0.0',
      install_requires=["autolab_core", "Box2D", "Cython", "dateutils", "h5py", "imageio", "keras", "numpy", "matplotlib", "opencv_python", "Pillow", "pybullet", "pygame", "scikit-image", "sklearn", "pyquaternion", "gym", "carbongym", "carbongym-utils", "GPy"],
      description='planorparam',
      author='Alex LaGrassa',
      author_email='lagrassa@cmu.edu',
      url='none',
      packages=['planning', 'modelfree', 'env', 'agent', 'real_robot', 'utils'],
     )
