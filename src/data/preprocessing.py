"""
Preprocessing and geometric calculation utilities for PathSnap.
All coordinates follow the dataset specification:
- nodes: [longitude, latitude]
- track: [longitude, latitude, timestamp]
"""

from typing import Tuple
import numpy as np


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Compute great-circle distance between coordinates in meters using Haversine formula.
    Supports scalars or numpy arrays.
    """
    earth_radius_m = 6371000.0  # WGS84 mean earth radius in meters

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return earth_radius_m * c


def compute_bearing(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Compute initial compass bearing (heading) from point 1 to point 2 in degrees [0, 360).
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    theta = np.degrees(np.arctan2(y, x))
    return (theta + 360.0) % 360.0


def bearing_difference(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Calculate minimal angular difference between two compass bearings in degrees [0, 180].
    """
    diff = np.abs(b1 - b2) % 360.0
    return np.minimum(diff, 360.0 - diff)


def point_to_segment_distance(
    p_lon: float,
    p_lat: float,
    a_lon: float,
    a_lat: float,
    b_lon: float,
    b_lat: float,
) -> Tuple[float, float, float, float]:
    """
    Project point P onto segment AB using local metric projection.
    
    Args:
        p_lon, p_lat: Coordinates of observation P
        a_lon, a_lat: Coordinates of segment start A
        b_lon, b_lat: Coordinates of segment end B
        
    Returns:
        perp_dist: Distance in meters from P to projected point on segment AB.
        t: Normalized projection scalar clamped to [0.0, 1.0].
        proj_lon: Longitude of projected point.
        proj_lat: Latitude of projected point.
    """
    mean_lat = (a_lat + b_lat) / 2.0
    lat_scale = 111139.0  # meters per degree latitude
    lon_scale = 111139.0 * np.cos(np.radians(mean_lat))

    dx = (b_lon - a_lon) * lon_scale
    dy = (b_lat - a_lat) * lat_scale
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-6:
        # Segment is degenerate (points coincide)
        dist = haversine_distance(p_lat, p_lon, a_lat, a_lon)
        return float(dist), 0.0, float(a_lon), float(a_lat)

    px = (p_lon - a_lon) * lon_scale
    py = (p_lat - a_lat) * lat_scale

    # Projection factor
    t_unclamped = (px * dx + py * dy) / seg_len_sq
    t = float(np.clip(t_unclamped, 0.0, 1.0))

    proj_x = a_lon * lon_scale + t * dx
    proj_y = a_lat * lat_scale + t * dy

    proj_lon = float(proj_x / lon_scale)
    proj_lat = float(proj_y / lat_scale)

    perp_dist = float(haversine_distance(p_lat, p_lon, proj_lat, proj_lon))
    return perp_dist, t, proj_lon, proj_lat


def clean_gps_track(track: np.ndarray, max_speed_mps: float = 60.0) -> np.ndarray:
    """
    Filter GPS outliers with unrealistic jumps or zero-displacement duplicate timestamps.
    track columns: [lon, lat, timestamp]
    """
    if len(track) <= 1:
        return track

    valid_mask = np.ones(len(track), dtype=bool)
    for i in range(1, len(track)):
        dt = track[i, 2] - track[i - 1, 2]
        if dt <= 0:
            dt = 1.0
        dist = haversine_distance(
            track[i, 1], track[i, 0], track[i - 1, 1], track[i - 1, 0]
        )
        speed = dist / dt
        if speed > max_speed_mps and dist > 150.0:
            valid_mask[i] = False

    cleaned = track[valid_mask]
    return cleaned if len(cleaned) > 0 else track
