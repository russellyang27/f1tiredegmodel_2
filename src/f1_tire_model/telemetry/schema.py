"""
Expected telemetry channel names, centralized so a typo in a column name
fails loudly in one place instead of silently producing NaN features three
modules downstream.

FastF1's `Lap.get_telemetry()` returns car channels merged with position
channels and an added `Distance` column. These constants name exactly the
subset this project relies on.
"""

from __future__ import annotations

# Car channels (from FastF1 car telemetry)
TIME = "Time"                  # timedelta since session start
SESSION_TIME = "SessionTime"
SPEED = "Speed"                 # kph
THROTTLE = "Throttle"           # 0-100
BRAKE = "Brake"                 # bool or 0-100 depending on FastF1 version
N_GEAR = "nGear"
RPM = "RPM"
DRS = "DRS"

# Position channels
X = "X"
Y = "Y"
Z = "Z"

# Derived (added by FastF1's add_distance() / get_telemetry())
DISTANCE = "Distance"

REQUIRED_CHANNELS: tuple[str, ...] = (TIME, SPEED, THROTTLE, BRAKE, N_GEAR, DISTANCE)

# Physically implausible bounds used to clip/flag corrupted samples.
# F1 cars do not exceed ~380 kph even on the longest straights (Monza/Baku),
# and cannot have negative speed.
MIN_PLAUSIBLE_SPEED_KPH = 0.0
MAX_PLAUSIBLE_SPEED_KPH = 380.0
