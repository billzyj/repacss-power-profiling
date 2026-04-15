"""Common exception types for the refactored architecture."""


class REPACSSPowerError(Exception):
    """Base exception for REPACSS power profiling."""


class CommentParseError(REPACSSPowerError):
    """Raised when a Slurm comment cannot be parsed structurally."""


class SlurmResolutionError(REPACSSPowerError):
    """Raised when Slurm job context cannot be resolved."""


class OOBBackendError(REPACSSPowerError):
    """Raised when an OOB backend fails."""


class InbandCollectorError(REPACSSPowerError):
    """Raised when an in-band collector fails."""
