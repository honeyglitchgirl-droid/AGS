"""Dimension-agnostic geometric field operators and PDE engines for AGS-Sci."""
from .operators import SpectralEngineND
from .geometry import (Metric, differential_form_derivative, exterior_derivative_1form,
                       lower_index, raise_index, contract, metric_gradient_norm,
                       box_scalar, tensor_trace, tensor_double_contract,
                       divergence_covariant_rank2)
from .turbulence2d import FastRFFTTurbulence2D, TurbulenceCheckpoint, TurbulenceRunResult

from .forced_turbulence2d import ForcedRFFTTurbulence2D, StationarityAudit, ForcedCheckpoint
from .turbulence3d import (
    PseudoSpectral3D, VectorDiagnostics3D, Simulation3DResult,
    low_storage_rk4, taylor_green, abc_flow, two_vortex_reconnection_seed, cfl_timestep,
    CK54_A, CK54_B, invariant_diagnostics_3d, taylor_green_dns_reference,
    anti_parallel_vortex_seed,
)
from .referee3d import PhysicalReferee3D, PhysicalGate3DResult, BKMMonitorResult
from .spherical_shell3d import SphericalShellToroidalDiffusion, SphericalShellResolution, chebyshev_lobatto, stretched_radius
from .spectral_estimators import negative_sobolev_norm, vector_negative_sobolev_norm, spectral_residual_loss
from .hybrid_spectral3d import FourierChebyshev3D, HybridGrid3D
from .linear_solvers import LinearSolveResult, gmres, polynomial_p_multigrid_preconditioner
from .spectral_refiner import RefinementResult, pde_residual_loss, spectral_gradient, refine_gradient_descent, STFNO3D
from .contracts import FieldCapabilities, ExperimentBudget, FieldSnapshot, FieldEngine
from .diagnostics import l2, linf, normalized_residual, normalized_divergence, kinetic_energy, energy_rate
from .registry import BackendSpec, FieldBackendRegistry, default_field_registry

from .four_d import FourDScalarFieldEngine, FourDScalarDiagnostics
from .five_d import (FiveDScalarFieldEngine, FiveDVectorFieldEngine,
                    FiveDScalarDiagnostics, FiveDVectorDiagnostics)
