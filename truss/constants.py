"""Global constants for the truss/bridge optimizer.

Units: length mm, force N, stress MPa (N/mm^2), modulus MPa, mass g,
volume mm^3.
"""

# Material: PLA (conservative values for FDM)
E = 2800.0          # MPa  (Young's modulus)
NU = 0.35           # Poisson's ratio
RHO_PLA = 1.24e-3   # g/mm^3 (density)
SIGMA_ALLOW = 20.0  # MPa  (allowable stress, 30 MPa / 1.5 SF)
SAFETY_FACTOR = 1.5

# Spec geometry / loading
SPAN = 500.0        # mm
MAX_DEFLECTION = 20.0   # mm
MAX_MASS = 150.0        # g
MIN_WIDTH = 4.0         # mm (narrowest member)
LOAD_PATCH = 15.0       # mm (center load spread)
SUPPORT_PATCH = 15.0    # mm (bearing at supports)

# Design-domain sweep ranges
D_RANGE = (40.0, 120.0)  # mm (truss depth)
H_RANGE = (3.0, 8.0)     # mm (constant out-of-plane height)

# SIMP optimization
SIMP_P = 3.0
FILTER_RMIN = 2.0     # mm
