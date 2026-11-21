"""Global constants for the truss/bridge optimizer.

Units: length mm, force N, stress MPa (N/mm^2), modulus MPa, mass g,
volume mm^3.
"""

# Material: PLA (FDM, loaded in-plane along the filament)
E = 2800.0          # MPa  (Young's modulus)
NU = 0.35           # Poisson's ratio
RHO_PLA = 1.24e-3   # g/mm^3 (density)
SIGMA_ALLOW = 35.0  # MPa  (in-plane PLA, no safety factor — competition is destructive)
SAFETY_FACTOR = 1.5  # kept for Euler buckling only (imperfection sensitivity)

# Spec geometry / loading
SPAN = 500.0        # mm
MAX_DEFLECTION = 20.0   # mm
MAX_MASS = 500.0        # g  (whole bridge: 2 trusses + deck + bracing)
MIN_WIDTH = 4.0         # mm (narrowest member)
LOAD_PATCH = 60.0       # mm (piston is 60 x 60 mm)
SUPPORT_PATCH = 15.0    # mm (bearing at supports)

# Design-domain sweep ranges
D_RANGE = (60.0, 200.0)  # mm (truss depth)
H_RANGE = (7.0, 7.0)     # mm (fixed out-of-plane height for final shape)

# Transverse bracing
BRACE_SPACING = 50.0    # mm (lateral-support spacing for out-of-plane buckling)

# SIMP optimization
SIMP_P = 3.0
FILTER_RMIN = 2.0     # mm
