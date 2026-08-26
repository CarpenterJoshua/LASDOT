"""
Acceptance tests for cross_section.

Run with:  pytest test_cross_section.py -v

Each test builds a small synthetic cloud, runs it through the real sorter to
get a real sorted cloud and index CSV, then cuts a section out of it. The
questions being answered are: does the section frame point the way it is
supposed to, does the clip honour the window it was given, do the recorded
parameters really put points back where they came from, and does anything get
lost on the way through.
"""

import os

import laspy
import numpy as np

from cross_section import (
    get_candidate_points,
    rotate_to_section_frame,
    select_section_window,
    write_cross_section,
)
from sort_by_station import sort_las_by_station
from station_projection import load_centerline, prepare_centerline
from test_sort_by_station import write_centerline, write_las


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# A centerline running due EAST. Travel is +x, so the RIGHT of travel is south,
# i.e. decreasing y. A point at (150, -10) is therefore 10 ft right of the road.
EAST = np.array([[0.0, 0.0], [300.0, 0.0]])

# A centerline with a right bend at (100, 0) and a left bend at (200, -50).
BENDS = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, -50.0], [300.0, -50.0]])


def build_sorted_cloud(tmp_path, centerline=EAST, count=8000, spread=25.0,
                       along=(0.0, 300.0), road_width=60.0, seed=0):
    """
    A synthetic cloud scattered along a centerline, already sorted by station.

    Returns (source_las, sorted_las, index_csv, centerline_path) -- everything a
    section needs, produced by the real sorter rather than faked.
    """
    rng = np.random.default_rng(seed)
    xy = np.column_stack(
        [
            rng.uniform(along[0], along[1], count),
            rng.uniform(-spread, spread, count),
        ]
    )
    source_las = str(tmp_path / "cloud.las")
    write_las(source_las, xy)

    # A real survey cloud carries a projection VLR, and the sorted cloud keeps
    # it. Add one so the section can be checked for correctly dropping it.
    las = laspy.read(source_las)
    las.header.vlrs.append(
        laspy.VLR(
            user_id="LASF_Projection",
            record_id=2112,
            description="fake wkt",
            record_data=b'PROJCS["fake"]\x00',
        )
    )
    las.write(source_las)

    centerline_path = write_centerline(tmp_path / "cl.txt", centerline)

    sorted_las = str(tmp_path / "cloud_SORTED.las")
    index_csv = str(tmp_path / "cloud_SORTED.csv")
    sort_las_by_station(
        source_las, centerline_path, road_width, sorted_las, index_csv,
        chunk_size=2000,
    )
    return source_las, sorted_las, index_csv, centerline_path


def read_transform_txt(path):
    """Read the `key = value` parameter file back into a dict."""
    params = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        params[key.strip()] = value.strip()
    return params


# --------------------------------------------------------------------------
# 1. the section frame
# --------------------------------------------------------------------------

def test_x_is_positive_to_the_right_of_travel(tmp_path):
    centerline = prepare_centerline(EAST)

    # Travelling east, so south (negative northing) is the right-hand side.
    right_of_road = np.array([[150.0, -10.0, 100.0]])
    left_of_road = np.array([[150.0, 10.0, 100.0]])

    rotated_right, _, _ = rotate_to_section_frame(right_of_road, centerline, 150.0)
    rotated_left, _, _ = rotate_to_section_frame(left_of_road, centerline, 150.0)

    assert rotated_right[0, 0] > 0.0
    assert rotated_left[0, 0] < 0.0
    np.testing.assert_allclose(rotated_right[0, 0], 10.0, atol=1e-9)
    np.testing.assert_allclose(rotated_left[0, 0], -10.0, atol=1e-9)


def test_y_runs_along_the_direction_of_travel(tmp_path):
    centerline = prepare_centerline(EAST)

    # 12 ft further east is 12 ft further along the road.
    ahead = np.array([[162.0, 0.0, 100.0]])
    rotated, _, _ = rotate_to_section_frame(ahead, centerline, 150.0)

    np.testing.assert_allclose(rotated[0], [0.0, 12.0, 100.0], atol=1e-9)


def test_the_frame_is_right_handed(tmp_path):
    centerline = prepare_centerline(BENDS)
    _, origin, _ = rotate_to_section_frame(np.zeros((1, 3)), centerline, 130.0)

    # Send the three survey axes through the rotation and read off where they
    # land. Those three images ARE the rotation matrix.
    basis = np.array([
        [origin[0] + 1.0, origin[1], 0.0],
        [origin[0], origin[1] + 1.0, 0.0],
        [origin[0], origin[1], 1.0],
    ])
    mapped, _, _ = rotate_to_section_frame(basis, centerline, 130.0)

    # A determinant of +1 is a pure rotation: no reflection, no scaling. -1
    # would mean the frame was flipped and left/right had swapped.
    np.testing.assert_allclose(np.linalg.det(mapped), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.cross(mapped[0], mapped[1]), mapped[2], atol=1e-9)


def test_the_rotation_is_rigid(tmp_path):
    rng = np.random.default_rng(4)
    points = np.column_stack([
        rng.uniform(80.0, 220.0, 200),
        rng.uniform(-60.0, 30.0, 200),
        rng.uniform(90.0, 110.0, 200),
    ])
    centerline = prepare_centerline(BENDS)
    rotated, _, _ = rotate_to_section_frame(points, centerline, 150.0)

    # Every pairwise distance must survive a rigid motion untouched.
    before = np.linalg.norm(points[:50, None, :] - points[None, :50, :], axis=-1)
    after = np.linalg.norm(rotated[:50, None, :] - rotated[None, :50, :], axis=-1)
    np.testing.assert_allclose(before, after, atol=1e-8)


def test_elevation_is_never_touched(tmp_path):
    rng = np.random.default_rng(5)
    points = np.column_stack([
        rng.uniform(0.0, 300.0, 100),
        rng.uniform(-40.0, 40.0, 100),
        rng.uniform(100.0, 200.0, 100),
    ])
    rotated, _, _ = rotate_to_section_frame(points, prepare_centerline(BENDS), 150.0)
    np.testing.assert_array_equal(rotated[:, 2], points[:, 2])


def test_section_x_matches_the_stored_offset_on_a_straight_run(tmp_path):
    """
    On a straight stretch the section's x axis and the cloud's stored `offset`
    are the same measurement, so they must agree exactly. This is the guard on
    the sign convention: if kappa flipped, every x would come back negated.
    """
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)
    station = 150.25

    candidates, _ = get_candidate_points(sorted_las, index_csv, station, 20.0)
    survey = np.column_stack([
        np.asarray(candidates.x), np.asarray(candidates.y), np.asarray(candidates.z)
    ])
    centerline = prepare_centerline(load_centerline(centerline_path))
    rotated, _, _ = rotate_to_section_frame(survey, centerline, station)

    np.testing.assert_allclose(
        rotated[:, 0], np.asarray(candidates.offset), atol=1e-8
    )
    np.testing.assert_allclose(
        rotated[:, 1], np.asarray(candidates.station) - station, atol=1e-8
    )


# --------------------------------------------------------------------------
# 2. the clip
# --------------------------------------------------------------------------

def test_clip_respects_left_right_and_thickness(tmp_path):
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)

    left, right, thickness = 5.0, 15.0, 2.0
    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=right, thickness=thickness, left_offset=left,
    )

    section = laspy.read(result["las_path"])
    x, y = np.asarray(section.x), np.asarray(section.y)

    assert x.min() >= -left
    assert x.max() <= right
    assert np.abs(y).max() <= thickness / 2.0

    # Asymmetric limits must actually be asymmetric -- points really do reach
    # further right than left.
    assert x.max() > left


def test_nothing_inside_the_window_is_missed(tmp_path):
    """The clip must be exactly the box, not a subset of it."""
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)
    station, left, right, thickness = 150.0, 20.0, 20.0, 1.0

    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=station, right_offset=right, thickness=thickness,
    )

    # Work out the answer independently from the whole candidate band.
    candidates, _ = get_candidate_points(sorted_las, index_csv, station, 20.0)
    survey = np.column_stack([
        np.asarray(candidates.x), np.asarray(candidates.y), np.asarray(candidates.z)
    ])
    centerline = prepare_centerline(load_centerline(centerline_path))
    rotated, _, _ = rotate_to_section_frame(survey, centerline, station)
    expected = select_section_window(rotated, left, right, thickness)

    assert result["points_kept"] == int(expected.sum())


def test_left_defaults_to_right(tmp_path):
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)

    implied = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=12.0, thickness=1.0,
    )
    spelled_out = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=12.0, thickness=1.0, left_offset=12.0,
    )
    assert implied["points_kept"] == spelled_out["points_kept"]

    x = np.asarray(laspy.read(implied["las_path"]).x)
    assert x.min() >= -12.0 and x.max() <= 12.0


# --------------------------------------------------------------------------
# 3. the transform parameters really do invert the section
# --------------------------------------------------------------------------

def test_txt_parameters_map_points_back_to_global(tmp_path):
    source_las, sorted_las, index_csv, centerline_path = build_sorted_cloud(
        tmp_path, centerline=BENDS
    )
    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=20.0, thickness=2.0,
    )

    params = read_transform_txt(result["txt_path"])
    origin_e = float(params["origin_easting"])
    origin_n = float(params["origin_northing"])
    cos_k = float(params["cos_kappa"])
    sin_k = float(params["sin_kappa"])

    section = laspy.read(result["las_path"])
    x, y = np.asarray(section.x), np.asarray(section.y)

    # The inverse, exactly as the .txt spells it out.
    easting = origin_e + x * cos_k - y * sin_k
    northing = origin_n + x * sin_k + y * cos_k

    # Compare against where those points actually live in the source cloud.
    source = laspy.read(source_las)
    came_from = np.asarray(section.orig_index).astype(np.int64)
    np.testing.assert_allclose(easting, np.asarray(source.x)[came_from], atol=1e-3)
    np.testing.assert_allclose(northing, np.asarray(source.y)[came_from], atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(section.z), np.asarray(source.z)[came_from], atol=1e-6
    )


def test_txt_records_the_window_that_was_asked_for(tmp_path):
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)
    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.25, right_offset=15.0, thickness=2.0, left_offset=7.5,
    )

    params = read_transform_txt(result["txt_path"])
    assert float(params["station"]) == 150.25
    assert float(params["right_offset"]) == 15.0
    assert float(params["left_offset"]) == 7.5
    assert float(params["thickness"]) == 2.0
    assert int(params["point_count"]) == result["points_kept"]
    assert params["centerline"] == centerline_path


# --------------------------------------------------------------------------
# 4. nothing is lost on the way through
# --------------------------------------------------------------------------

def test_all_fields_survive(tmp_path):
    source_las, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)
    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=20.0, thickness=2.0,
    )

    source = laspy.read(source_las)
    section = laspy.read(result["las_path"])
    came_from = np.asarray(section.orig_index).astype(np.int64)

    for name in ("intensity", "classification", "red", "return_number", "gps_time"):
        np.testing.assert_array_equal(
            np.asarray(section[name]), np.asarray(source[name])[came_from],
            err_msg=f"{name} did not survive the section",
        )

    # The sorted cloud's own extra dims come along too.
    names = [dim.name for dim in section.point_format.extra_dimensions]
    assert set(names) == {"orig_index", "station", "offset"}


def test_the_section_does_not_claim_the_survey_crs(tmp_path):
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)
    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=20.0, thickness=1.0,
    )

    # The sorted cloud is in the survey CRS; the section is in local feet, so
    # it must not carry the projection forward.
    assert any(v.user_id == "LASF_Projection" for v in laspy.read(sorted_las).header.vlrs)
    header = laspy.read(result["las_path"]).header
    assert not any(v.user_id == "LASF_Projection" for v in header.vlrs)

    params = read_transform_txt(result["txt_path"])
    assert "LASF_Projection" in params["source_crs_vlr"]


# --------------------------------------------------------------------------
# 5. the read stays small
# --------------------------------------------------------------------------

def test_only_the_indexed_range_is_read(tmp_path):
    _, sorted_las, index_csv, _ = build_sorted_cloud(tmp_path, along=(0.0, 300.0))
    station, half_length = 150.0, 20.0

    candidates, _ = get_candidate_points(sorted_las, index_csv, station, half_length)

    with laspy.open(sorted_las) as reader:
        total = reader.header.point_count

    # A 40 ft band out of a 300 ft road: a fraction of the file, not all of it.
    assert len(candidates) < total / 3

    # And every point read really is from that band. The index is granular to
    # the foot, so allow one foot of slop either side.
    station_values = np.asarray(candidates.station)
    assert station_values.min() >= station - half_length - 1.0
    assert station_values.max() <= station + half_length + 1.0


# --------------------------------------------------------------------------
# 6. output naming
# --------------------------------------------------------------------------

def test_station_filename_uses_padded_feet_and_hundredths(tmp_path):
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)

    partial = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.25, right_offset=20.0, thickness=1.0,
    )
    assert os.path.basename(partial["las_path"]) == "cloud_SORTED_sta000150_25.las"
    assert os.path.basename(partial["txt_path"]) == "cloud_SORTED_sta000150_25.txt"

    whole = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=100.0, right_offset=20.0, thickness=1.0,
    )
    assert os.path.basename(whole["las_path"]) == "cloud_SORTED_sta000100_00.las"


def test_large_stations_fill_all_six_digits(tmp_path):
    """A station of 15025.00 ft must read sta015025_00, not overflow the pad."""
    long_road = np.array([[0.0, 0.0], [20000.0, 0.0]])
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(
        tmp_path, centerline=long_road, count=4000, along=(15000.0, 15050.0),
    )

    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=15025.0, right_offset=20.0, thickness=1.0,
    )
    assert os.path.basename(result["las_path"]) == "cloud_SORTED_sta015025_00.las"


def test_out_dir_redirects_both_files(tmp_path):
    _, sorted_las, index_csv, centerline_path = build_sorted_cloud(tmp_path)
    elsewhere = tmp_path / "sections"
    elsewhere.mkdir()

    result = write_cross_section(
        sorted_las, index_csv, centerline_path,
        station=150.0, right_offset=20.0, thickness=1.0, out_dir=str(elsewhere),
    )
    assert os.path.dirname(result["las_path"]) == str(elsewhere)
    assert os.path.exists(result["las_path"])
    assert os.path.exists(result["txt_path"])
