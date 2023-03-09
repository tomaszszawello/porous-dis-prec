# porous-dis-prec
Package for simulating dissolution and precipitation in porous media

WARNING: use Python 3.10 to have type hints etc. working properly. I also
recommend working with VS Code (or simillar) to make use of Pylance extension
(which e.g. supports docstrings for class attributes).

Necessary packages:
dill
matplotlib
networkx
numpy
scipy

Set simulation parameters in config, then start __main__ and wait for wonderful
results to appear in the simulation directory. Initialization may take a while
(~1 min for network size 100). Simulation should work fine up to network size
200, above it may need large RAM or take very long.

New version with volume tracking and nucleation/passivation processes is under
construction. I may update this one with a few utilities like saving/loading in
VTK format or better data collection.

Should you have any questions, contact me at tomasz.szawello@fuw.edu.pl.

Enjoy!
