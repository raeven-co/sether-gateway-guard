"""Build the Sether detector set from settings."""

from collections.abc import Iterable, Sequence

from sether import (
    Detector,
    basic_detectors,
    create_phone_detector,
    identity_detectors,
    secrets_detectors,
)

from .settings import DetectorPack


def build_detectors(
    packs: Iterable[DetectorPack], phone_region: str | None
) -> tuple[Detector, ...]:
    selected = set(packs)
    detectors: list[Detector] = []
    if "basic" in selected:
        detectors.extend(_basic_detectors(phone_region))
    if "secrets" in selected:
        detectors.extend(secrets_detectors)
    if "identity" in selected:
        detectors.extend(identity_detectors)
    return tuple(detectors)


def detector_types(detectors: Sequence[Detector]) -> frozenset[str]:
    return frozenset(detector.type for detector in detectors)


def _basic_detectors(phone_region: str | None) -> list[Detector]:
    if phone_region is None:
        return list(basic_detectors)
    without_phone = [detector for detector in basic_detectors if detector.type != "PHONE"]
    return [*without_phone, create_phone_detector(default_country=phone_region)]
