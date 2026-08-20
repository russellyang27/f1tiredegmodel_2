"""Custom exceptions for the telemetry module.

A dedicated exception type (instead of letting raw FastF1/pandas exceptions
propagate) lets callers upstream (feature extractors, dataset builders)
catch "this lap's telemetry just isn't usable" as a single, expected case —
e.g. to skip the lap and log it, rather than crashing a multi-hour dataset
build over one bad lap.
"""


class TelemetryNotAvailableError(Exception):
    """Raised when telemetry for a requested lap cannot be retrieved or is unusable
    (e.g. empty, missing required channels after cleaning)."""
