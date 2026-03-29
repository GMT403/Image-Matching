import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def parse_dt(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_float(text):
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except ValueError:
        return None


@dataclass
class FlightNavPoint:
    time_utc: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    yaw_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    gps_speed_mps: Optional[float] = None
    gps_course_deg: Optional[float] = None
    imu_gyro_magnitude: Optional[float] = None


class FlightLogNavigation:
    def __init__(
        self,
        ahr_points: list[FlightNavPoint],
        gps_points: list[FlightNavPoint],
        imu_points: list[FlightNavPoint],
    ) -> None:
        self.ahr_points = sorted(ahr_points, key=lambda item: item.time_utc)
        self.gps_points = sorted(gps_points, key=lambda item: item.time_utc)
        self.imu_points = sorted(imu_points, key=lambda item: item.time_utc)

    @classmethod
    def from_log_folder(cls, log_folder: str) -> "FlightLogNavigation":
        ahr_points = cls._load_ahr_points(os.path.join(log_folder, "AHR2.csv"))
        gps_points = cls._load_gps_points(os.path.join(log_folder, "GPS.csv"))
        imu_points = cls._load_imu_points(os.path.join(log_folder, "IMU.csv"))
        return cls(ahr_points=ahr_points, gps_points=gps_points, imu_points=imu_points)

    @staticmethod
    def _load_ahr_points(csv_path: str) -> list[FlightNavPoint]:
        points = []
        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                time_utc = parse_dt(row.get("UTC_Approx") or row.get("UTC_GPS"))
                if time_utc is None:
                    continue
                points.append(
                    FlightNavPoint(
                        time_utc=time_utc,
                        latitude=parse_float(row.get("Lat")),
                        longitude=parse_float(row.get("Lng") or row.get("Lon")),
                        altitude_m=parse_float(row.get("Alt")),
                        yaw_deg=parse_float(row.get("Yaw")),
                        roll_deg=parse_float(row.get("Roll")),
                        pitch_deg=parse_float(row.get("Pitch")),
                    )
                )
        return points

    @staticmethod
    def _load_gps_points(csv_path: str) -> list[FlightNavPoint]:
        points = []
        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                time_utc = parse_dt(row.get("UTC_GPS") or row.get("UTC_Approx"))
                if time_utc is None:
                    continue
                points.append(
                    FlightNavPoint(
                        time_utc=time_utc,
                        latitude=parse_float(row.get("Lat")),
                        longitude=parse_float(row.get("Lng") or row.get("Lon")),
                        altitude_m=parse_float(row.get("Alt")),
                        gps_speed_mps=parse_float(row.get("Spd")),
                        gps_course_deg=parse_float(row.get("GCrs") or row.get("Heading") or row.get("Yaw")),
                    )
                )
        return points

    @staticmethod
    def _load_imu_points(csv_path: str) -> list[FlightNavPoint]:
        points = []
        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                time_utc = parse_dt(row.get("UTC_Approx") or row.get("UTC_GPS"))
                gx = parse_float(row.get("GyrX"))
                gy = parse_float(row.get("GyrY"))
                gz = parse_float(row.get("GyrZ"))
                if time_utc is None or gx is None or gy is None or gz is None:
                    continue
                points.append(
                    FlightNavPoint(
                        time_utc=time_utc,
                        imu_gyro_magnitude=math.sqrt(gx * gx + gy * gy + gz * gz),
                    )
                )
        return points

    def at(self, time_utc: datetime) -> FlightNavPoint:
        ahr = self._interpolate_nav_points(self.ahr_points, time_utc)
        gps = self._interpolate_nav_points(self.gps_points, time_utc)
        imu = self._interpolate_nav_points(self.imu_points, time_utc)
        return FlightNavPoint(
            time_utc=time_utc,
            latitude=(ahr.latitude if ahr else None) or (gps.latitude if gps else None),
            longitude=(ahr.longitude if ahr else None) or (gps.longitude if gps else None),
            altitude_m=(ahr.altitude_m if ahr else None) or (gps.altitude_m if gps else None),
            yaw_deg=ahr.yaw_deg if ahr else None,
            roll_deg=ahr.roll_deg if ahr else None,
            pitch_deg=ahr.pitch_deg if ahr else None,
            gps_speed_mps=gps.gps_speed_mps if gps else None,
            gps_course_deg=gps.gps_course_deg if gps else None,
            imu_gyro_magnitude=imu.imu_gyro_magnitude if imu else None,
        )

    @classmethod
    def _interpolate_nav_points(cls, points: list[FlightNavPoint], time_utc: datetime) -> Optional[FlightNavPoint]:
        if not points:
            return None
        if time_utc <= points[0].time_utc:
            return points[0]
        if time_utc >= points[-1].time_utc:
            return points[-1]

        lo = 0
        hi = len(points) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if points[mid].time_utc < time_utc:
                lo = mid + 1
            else:
                hi = mid - 1

        upper_index = max(1, lo)
        p0 = points[upper_index - 1]
        p1 = points[upper_index]
        total_seconds = (p1.time_utc - p0.time_utc).total_seconds()
        if total_seconds <= 0.0:
            return p0
        alpha = (time_utc - p0.time_utc).total_seconds() / total_seconds

        return FlightNavPoint(
            time_utc=time_utc,
            latitude=cls._interp_linear(p0.latitude, p1.latitude, alpha),
            longitude=cls._interp_linear(p0.longitude, p1.longitude, alpha),
            altitude_m=cls._interp_linear(p0.altitude_m, p1.altitude_m, alpha),
            yaw_deg=cls._interp_angle_deg(p0.yaw_deg, p1.yaw_deg, alpha),
            roll_deg=cls._interp_linear(p0.roll_deg, p1.roll_deg, alpha),
            pitch_deg=cls._interp_linear(p0.pitch_deg, p1.pitch_deg, alpha),
            gps_speed_mps=cls._interp_linear(p0.gps_speed_mps, p1.gps_speed_mps, alpha),
            gps_course_deg=cls._interp_angle_deg(p0.gps_course_deg, p1.gps_course_deg, alpha),
            imu_gyro_magnitude=cls._interp_linear(p0.imu_gyro_magnitude, p1.imu_gyro_magnitude, alpha),
        )

    @staticmethod
    def _interp_linear(v0, v1, alpha: float):
        if v0 is None and v1 is None:
            return None
        if v0 is None:
            return v1
        if v1 is None:
            return v0
        return float(v0) + (float(v1) - float(v0)) * float(alpha)

    @staticmethod
    def _interp_angle_deg(v0, v1, alpha: float):
        if v0 is None and v1 is None:
            return None
        if v0 is None:
            return v1
        if v1 is None:
            return v0
        delta = (float(v1) - float(v0) + 180.0) % 360.0 - 180.0
        return (float(v0) + delta * float(alpha)) % 360.0
