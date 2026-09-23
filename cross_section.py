"""
Cut a cross-section out of a station-sorted point cloud.

Give this a sorted cloud (from sort_by_station.py), its index CSV, the
centerline, and a station, and it writes the slice of road standing at that
station -- rotated out of survey coordinates into a local section frame:

    +y  along the centerline, in the direction of travel
    +x  square to it, increasing to the RIGHT of travel
    +z  elevation, exactly as it was -- never touched

which is right-handed, and has its origin sitting on the centerline at the
requested station. So a point's x IS its offset from the road centre and its y
is how far ahead of (or behind) the section it stands.

The rotation follows PointCloudSectioning.ipynb: kappa = -arctan2(v_x, v_y),
applied with the Rk matrix written the same way. Only kappa is used, because
the new x and y axes have to stay in the old xy plane.

Two files come out, both named for the station:

    <sorted stem>_sta000150_25.las   the section, every field preserved
    <sorted stem>_sta000150_25.txt   the numbers needed to put any section
                                     point back into survey coordinates

MEMORY. The cloud is never loaded. The index CSV says which rows hold the
stations near the one asked for, and only those rows are read -- one seek, one
contiguous read.
"""

import os
from copy import deepcopy

import laspy
import numpy as np

from station_projection import load_centerline, prepare_centerline


# --------------------------------------------------------------------------
# 1. get the candidate points out of the sorted cloud
# --------------------------------------------------------------------------

def get_candidate_points(sorted_las_path, index_csv_path, station, half_length):
    """
    Read the band of points whose stations surround the requested one.

    sorted_las_path : a cloud written by sort_by_station.py, so its rows are in
                      station order and it carries the `station` extra dim.
    index_csv_path  : the `station_ft,start_index` table written alongside it.
    station         : the station the section stands at. May be partial.
    half_length     : how far either side of it to gather, in feet.

    Returns (candidates, source_header).

    We deliberately gather MORE than the final section needs. Station is
    measured as arc length along a bending centerline, but the section frame is
    flat, so at a bend a point's station and its position along the section's
    y axis are not the same number. Grabbing a generous band and clipping later
    in the ROTATED frame gets the right answer either way; clipping on station
    would not.

    The read itself is the whole reason the cloud was sorted. Two binary
    searches in a table of a few hundred rows say which rows of the LAS hold
    this band, and those rows are one contiguous stretch -- so this is a single
    seek and a single read, no matter how big the file is.
    """
    table = np.loadtxt(index_csv_path, delimiter=",", skiprows=1, ndmin=2)
    feet = table[:, 0]
    starts = table[:, 1].astype(np.int64)

    low_station = station - half_length
    high_station = station + half_length

    # The row for the last foot at or below the bottom of the band. searchsorted
    # with side="right" gives the first foot ABOVE it, so step back one to get
    # the foot our band starts inside of. Clamped in case the band begins before
    # the first foot in the table.
    row_low = max(int(np.searchsorted(feet, low_station, side="right")) - 1, 0)

    # The row for the first foot PAST the top of the band. Its start_index is
    # where the band stops -- which is why build_foot_index appends a sentinel:
    # even a band running to the end of the cloud has a row to point at here.
    row_high = min(
        int(np.searchsorted(feet, high_station, side="right")), len(feet) - 1
    )

    start = int(starts[row_low])
    stop = int(starts[row_high])

    with laspy.open(sorted_las_path) as reader:
        source_header = reader.header
        reader.seek(start)
        candidates = reader.read_points(stop - start)

    return candidates, source_header


# --------------------------------------------------------------------------
# 2. rotate them into the section frame
# --------------------------------------------------------------------------

def rotate_to_section_frame(points_xyz, centerline, station):
    """
    Shift and rotate points so the section stands at the origin.

    points_xyz : (N, 3) survey coordinates -- easting, northing, elevation.
    centerline : a PreparedCenterline (from station_projection.prepare_centerline).
    station    : the station the section stands at.

    Returns (rotated_xyz, origin, kappa), where `origin` is the survey (easting,
    northing) of the centerline at that station and `kappa` is the rotation
    angle in radians. Both are needed to undo this later.

    Two steps, in order:

    THE SHIFT moves the centerline point at this station to (0, 0). Elevation
    is left alone -- the centerline is a 2D thing with no elevation of its own,
    and a section is far more useful reading true elevations off its z axis.

    THE ROTATION turns the cloud about the vertical axis until the direction of
    travel points along +y. Only kappa is needed: the section frame stays level,
    so the new x and y axes stay in the old xy plane, and omega and phi (the
    other two angles in PointCloudSectioning.ipynb) are both zero.

    This is a rigid motion -- it preserves every distance and angle. That is
    what separates it from station/offset, which stretches and squeezes around
    bends, and it is why the inverse recorded in the .txt is exact rather than
    approximate.
    """
    # WHICH SEGMENT owns this station? station0 holds the station at the start
    # of each segment and increases along the centerline, so a binary search
    # finds the first segment starting past our station -- and the one before
    # it is the segment we are standing on.
    segment = int(np.searchsorted(centerline.station0, station, side="right")) - 1
    segment = min(max(segment, 0), len(centerline.length) - 1)

    # Walk from that segment's start vertex, along its direction, however far
    # past the segment's own start station we are. That lands on the centerline
    # at exactly the requested station.
    along = station - centerline.station0[segment]
    origin = centerline.seg_start[segment] + along * centerline.direction[segment]

    # The pointing vector: which way the road runs here.
    v = centerline.direction[segment]

    # Shift. The third column is 0 because elevation is not moved.
    new_pts = points_xyz - np.array([origin[0], origin[1], 0.0])

    # Rotate. kappa is negated so that the vector v ends up lying along +y
    # rather than the rotation being read the other way round; the sign works
    # out so that a point to the right of travel gets a positive x.
    k = -np.arctan2(v[0], v[1])

    Rk = np.array([[ np.cos(k),  np.sin(k), 0],
                   [-np.sin(k),  np.cos(k), 0],
                   [         0,          0, 1]])

    new_pts = (Rk @ new_pts.T).T

    return new_pts, origin, k


# --------------------------------------------------------------------------
# 3. clip to the window that was actually asked for
# --------------------------------------------------------------------------

def select_section_window(rotated_xyz, left_offset, right_offset, thickness):
    """
    Which of the rotated points fall inside the requested section?

    rotated_xyz  : (N, 3) points already in the section frame.
    left_offset  : how far to keep to the LEFT of the centerline, in feet.
    right_offset : how far to keep to the RIGHT, in feet.
    thickness    : how deep the slab is along the road; the section keeps
                   half of it either side of the station.

    Returns a boolean mask, one entry per point -- NOT the points themselves.
    A mask is what the caller wants, because it can be applied to the whole
    laspy record and carry every field (colour, intensity, classification, GPS
    time, the extra dims) through the clip untouched.

    x is the across-road axis and is positive to the RIGHT of travel, so the
    right limit is +right_offset and the left limit is -left_offset. y is the
    along-road axis, zero at the station, so the slab is |y| <= half thickness.
    """
    x = rotated_xyz[:, 0]
    y = rotated_xyz[:, 1]

    inside_width = (x >= -left_offset) & (x <= right_offset)
    inside_slab = np.abs(y) <= thickness / 2.0

    return inside_width & inside_slab


# --------------------------------------------------------------------------
# the output files
# --------------------------------------------------------------------------

def _section_output_header(source_header):
    """
    Build the section's header from the sorted cloud's own.

    Everything is inherited -- point format, scales, and the extra dimensions
    (orig_index, station, offset), so a section point can still be traced back.
    Two things change:

    THE OFFSETS. LAS stores coordinates as integer counts of `scale` measured
    from `offset`. The section's x and y are now small local numbers around
    zero, so their offsets become 0. Z keeps the source's offset AND scale,
    which means the stored elevation integers copy across bit-exactly rather
    than being requantised.

    THE CRS. The section's x and y are local feet from the station origin, not
    survey coordinates, so carrying the source's projection VLRs forward would
    make the file lie about itself. They are stripped, and the .txt records
    what the parent CRS was along with the numbers to get back to it.
    """
    header = deepcopy(source_header)
    header.offsets = np.array([0.0, 0.0, source_header.offsets[2]], dtype=float)
    header.scales = np.asarray(source_header.scales, dtype=float)

    stripped = []
    for vlr in list(header.vlrs):
        if vlr.user_id == "LASF_Projection":
            stripped.append(f"{vlr.user_id} {vlr.record_id}")
            header.vlrs.remove(vlr)
    header.global_encoding.wkt = False

    return header, stripped


def _write_transform_txt(path, parameters):
    """
    Write the numbers needed to put section points back into survey coordinates.

    Plain `key = value` lines, then a comment block spelling out the inverse.
    cos and sin of kappa are written out too, so anything reading this file can
    do the arithmetic without calling a trig function.
    """
    with open(path, "w") as handle:
        handle.write("# Cross-section transform parameters\n")
        handle.write("# Written by cross_section.py\n\n")

        width = max(len(name) for name in parameters)
        for name, value in parameters.items():
            handle.write(f"{name:<{width}} = {value}\n")

        handle.write("\n")
        handle.write("# Section (x, y, z) back to survey (E, N, Z):\n")
        handle.write("#     E = origin_easting  + x*cos_kappa - y*sin_kappa\n")
        handle.write("#     N = origin_northing + x*sin_kappa + y*cos_kappa\n")
        handle.write("#     Z = z          (elevation was never altered)\n")
        handle.write("#\n")
        handle.write("# The section frame is right-handed: +x is right of travel,\n")
        handle.write("# +y is the direction of travel, +z is up.\n")


# --------------------------------------------------------------------------
# 4. the whole job, in one call
# --------------------------------------------------------------------------

def write_cross_section(sorted_las, index_csv, centerline_path, station, right_offset,
    thickness, left_offset=None, out_dir=None, half_length=None):
    """
    Cut one cross-section and write it, with its transform parameters, to disk.

    sorted_las      : a cloud written by sort_by_station.py.
    index_csv       : the `station_ft,start_index` table written with it.
    centerline_path : the same centerline the cloud was sorted against.
    station         : where the section stands. May be partial, e.g. 150.25.
    right_offset    : how far to keep to the right of the centerline, in feet.
    thickness       : how deep the slab is along the road, in feet.
    left_offset     : how far to keep to the left. Defaults to right_offset,
                      giving a section centred on the road.
    out_dir         : where to write. Defaults to beside the sorted cloud.
    half_length     : how much station to gather before clipping. Defaults to
                      half the section's width, which is generous enough to
                      survive a bend; pass a smaller number on a stretch you
                      know is straight and the read gets cheaper.

    Returns a dict: the two output paths, the transform parameters, and the
    counts (how many points were read, how many the window kept).

    Writes <sorted stem>_sta000150_25.las and .txt -- station 150.25 padded to
    six whole feet, with the hundredths after an underscore.
    """
    if left_offset is None:
        left_offset = right_offset

    if half_length is None:
        # A band about as long as the section is wide. Floored at half the slab
        # depth so a request for a very thick section still gathers enough.
        half_length = max((left_offset + right_offset) / 2.0, thickness / 2.0)

    centerline = prepare_centerline(load_centerline(centerline_path))

    # 1: pull the band of candidates out of the sorted cloud.
    candidates, source_header = get_candidate_points(
        sorted_las, index_csv, station, half_length
    )
    if len(candidates) == 0:
        raise ValueError(
            f"no points near station {station:g} -- check that the station lies "
            "inside the sorted cloud's range"
        )

    survey_xyz = np.column_stack(
        [
            np.asarray(candidates.x),
            np.asarray(candidates.y),
            np.asarray(candidates.z),
        ]
    )

    # 2: move them into the section frame.
    rotated, origin, kappa = rotate_to_section_frame(survey_xyz, centerline, station)

    # 3: keep only what was asked for.
    keep = select_section_window(rotated, left_offset, right_offset, thickness)
    if not keep.any():
        raise ValueError(
            f"the window at station {station:g} is empty -- no points within "
            f"{left_offset:g} left / {right_offset:g} right and "
            f"{thickness:g} thick"
        )

    # Name both outputs for the station: six whole feet, then the hundredths.
    # Rounding to hundredths FIRST keeps 150.999 from producing three digits
    # after the underscore.
    hundredths_total = int(round(station * 100))
    whole_feet, hundredths = divmod(hundredths_total, 100)
    suffix = f"sta{whole_feet:06d}_{hundredths:02d}"

    stem, _ = os.path.splitext(sorted_las)
    if out_dir is not None:
        stem = os.path.join(out_dir, os.path.basename(stem))
    out_las = f"{stem}_{suffix}.las"
    out_txt = f"{stem}_{suffix}.txt"

    # Copy the survivors into a record matching the OUTPUT point format. Same
    # reasoning as build_sorted_record in sort_by_station.py: copying `.array`
    # fields by name moves the packed bytes across wholesale, so bit-packed
    # dimensions and the stored Z integers are not reinterpreted on the way.
    header, stripped_vlrs = _section_output_header(source_header)
    kept = candidates[keep]
    out = laspy.ScaleAwarePointRecord.zeros(len(kept), header=header)
    for name in kept.array.dtype.names:
        out.array[name] = kept.array[name]

    # Assigning x/y stores (value - offset) / scale using the OUTPUT offsets,
    # which are zero -- so the local section coordinates land correctly. z is
    # deliberately left as copied; see _section_output_header.
    out.x = rotated[keep, 0]
    out.y = rotated[keep, 1]

    with laspy.open(out_las, mode="w", header=header) as writer:
        writer.write_points(out)

    parameters = {
        "source_las": sorted_las,
        "source_index": index_csv,
        "centerline": centerline_path,
        "source_crs_vlr": (
            ", ".join(stripped_vlrs) + " (stripped from this section)"
            if stripped_vlrs
            else "none in source"
        ),
        "station": f"{station:.4f}",
        "left_offset": f"{left_offset:.4f}",
        "right_offset": f"{right_offset:.4f}",
        "thickness": f"{thickness:.4f}",
        "origin_easting": f"{origin[0]:.4f}",
        "origin_northing": f"{origin[1]:.4f}",
        "kappa_rad": f"{kappa:.12f}",
        "kappa_deg": f"{np.degrees(kappa):.6f}",
        "cos_kappa": f"{np.cos(kappa):.12f}",
        "sin_kappa": f"{np.sin(kappa):.12f}",
        "point_count": str(len(kept)),
    }
    _write_transform_txt(out_txt, parameters)

    return {
        "las_path": out_las,
        "txt_path": out_txt,
        "origin": (float(origin[0]), float(origin[1])),
        "kappa": float(kappa),
        "candidates_read": len(candidates),
        "points_kept": int(len(kept)),
    }
