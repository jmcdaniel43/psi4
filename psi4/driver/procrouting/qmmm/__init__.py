"""
A helper folder for qm/mm functions
"""

_have_qmmm = False

try:
   from . import grid_interface
   _have_qmmm = True
except ImportError:
   pass

