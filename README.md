This is the repo for my passive bistatic radar sim.
The standalone script contains all of the salient functions, and requires the following libraries (aside from the standard Anaconda distro):
1. netCDF4
2. numexpr (necessary for parallelization)
3. numba (not *actually* necessary, but greatly speeds up some functions)
4. pytmatrix (optional for simulations with T-matrix scattering enabled)
