"""
Acceptance tests for sort_by_station.

Run with:  pytest test_sort_by_station.py -v

The question these answer is the one the whole script exists for: can you take
a station window, look its rows up in the CSV, read that one contiguous range
out of the sorted LAS, and get back exactly the points in that window -- with
every attribute intact?
"""

import laspy
import numpy as np

from sort_by_station import build_foot_index, sort_las_by_station


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# A polyline with a right bend at (100, 0) and a left bend at (200, -50),
# travelling broadly east. Same shape the projection tests use.
BENDS = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, -50.0], [300.0, -50.0]])


def write_centerline(path, vertices):
    """Write a 2D vertex list in the comma-separated form load_centerline reads."""
    lines = [f"{x},{y}" for x, y in vertices]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def write_las(path, xy, z=None):
    """
    A small synthetic LAS with point format 3 (RGB + GPS time).

    Every attribute is given a distinct per-point value derived from the row
    number, so after the reorder we can check that row R still carries the
    values its original row had.
    """
    xy = np.asarray(xy, dtype=float)
    count = len(xy)
    z = np.linspace(100.0, 200.0, count) if z is None else np.asarray(z, dtype=float)

    header = laspy.LasHeader(version="1.4", point_format=3)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([xy[:, 0].min(), xy[:, 1].min(), z.min()])

    las = laspy.LasData(header)
    las.x, las.y, las.z = xy[:, 0], xy[:, 1], z
    las.intensity = np.arange(count, dtype=np.uint16) % 65535
    las.classification = (np.arange(count) % 31).astype(np.uint8)
    las.red = (np.arange(count) % 65535).astype(np.uint16)
    las.gps_time = np.arange(count, dtype=np.float64) * 0.5
    las.return_number = (np.arange(count) % 5 + 1).astype(np.uint8)
    las.write(str(path))
    return las


def read_foot_index(csv_path):
    """Read the index CSV back as (foot_stations, start_indices)."""
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    return table[:, 0], table[:, 1].astype(np.int64)


def build_corridor_cloud(tmp_path, count=4000, half_spread=60.0, seed=0):
    """
    A cloud scattered around BENDS, plus a matching centerline file.

    `half_spread` is deliberately wider than the road widths the tests use, so
    there is always something for the corridor cull to throw away.
    """
    rng = np.random.default_rng(seed)
    xy = np.column_stack(
        [
            rng.uniform(-20.0, 320.0, count),
            rng.uniform(-half_spread - 60.0, half_spread, count),
        ]
    )
    las_path = tmp_path / "cloud.las"
    write_las(las_path, xy)
    centerline_path = write_centerline(tmp_path / "cl.txt", BENDS)
    return str(las_path), centerline_path


def run_sort(tmp_path, road_width=40.0, foot_step=1.0, chunk_size=500, **kwargs):
    """Build both outputs and hand back their paths plus the counts."""
    las_path, centerline_path = build_corridor_cloud(tmp_path, **kwargs)
    out_las = str(tmp_path / "sorted.las")
    out_csv = str(tmp_path / "index.csv")
    counts = sort_las_by_station(
        las_path,
        centerline_path,
        road_width,
        out_las,
        out_csv,
        foot_step=foot_step,
        chunk_size=chunk_size,
    )
    return las_path, out_las, out_csv, counts


# --------------------------------------------------------------------------
# 1. the whole point: a station window is one contiguous read
# --------------------------------------------------------------------------

def test_station_window_reads_as_one_contiguous_range(tmp_path):
    _, out_las, out_csv, _ = run_sort(tmp_path)
    feet, starts = read_foot_index(out_csv)

    low_foot, high_foot = 120.0, 140.0
    # The window's first row, and the row where the foot PAST the window
    # begins -- which is the exclusive end of the window.
    start = int(starts[np.searchsorted(feet, low_foot)])
    stop = int(starts[np.searchsorted(feet, high_foot + 1.0)])
    assert stop > start, "test window should not be empty"

    with laspy.open(out_las) as reader:
        reader.seek(start)
        window = reader.read_points(stop - start)
    station = np.asarray(window.station)

    # Everything read back is inside the window...
    assert station.min() >= low_foot
    assert station.max() < high_foot + 1.0

    # ...and nothing inside the window was left outside the range.
    with laspy.open(out_las) as reader:
        everything = np.asarray(reader.read().station)
    in_window = (everything >= low_foot) & (everything < high_foot + 1.0)
    assert int(in_window.sum()) == len(station)


def test_output_is_sorted_by_station(tmp_path):
    _, out_las, _, counts = run_sort(tmp_path)
    with laspy.open(out_las) as reader:
        station = np.asarray(reader.read().station)

    assert len(station) == counts["kept"]
    assert np.all(np.diff(station) >= 0.0), "output is not in station order"


def test_every_foot_row_points_at_its_own_foot(tmp_path):
    """Walk the whole index, not just one window, and check each bin."""
    _, out_las, out_csv, counts = run_sort(tmp_path)
    feet, starts = read_foot_index(out_csv)

    with laspy.open(out_las) as reader:
        station = np.asarray(reader.read().station)

    # The sentinel closes the table: its start is one past the last real row.
    assert starts[-1] == counts["kept"] == len(station)

    for i in range(len(feet) - 1):
        block = station[starts[i]:starts[i + 1]]
        if len(block) == 0:
            continue          # a foot with no points -- allowed, see below
        assert block.min() >= feet[i]
        assert block.max() < feet[i] + 1.0


# --------------------------------------------------------------------------
# 2. the reorder must not disturb the points themselves
# --------------------------------------------------------------------------

def test_attributes_survive_the_reorder(tmp_path):
    source_las, out_las, _, _ = run_sort(tmp_path)

    source = laspy.read(source_las)
    sorted_cloud = laspy.read(out_las)

    # orig_index says which source row each output row came from; every
    # attribute should match that row exactly.
    came_from = np.asarray(sorted_cloud.orig_index).astype(np.int64)

    np.testing.assert_array_equal(
        np.asarray(sorted_cloud.intensity), np.asarray(source.intensity)[came_from]
    )
    np.testing.assert_array_equal(
        np.asarray(sorted_cloud.classification),
        np.asarray(source.classification)[came_from],
    )
    np.testing.assert_array_equal(
        np.asarray(sorted_cloud.red), np.asarray(source.red)[came_from]
    )
    np.testing.assert_array_equal(
        np.asarray(sorted_cloud.return_number),
        np.asarray(source.return_number)[came_from],
    )
    np.testing.assert_array_equal(
        np.asarray(sorted_cloud.gps_time), np.asarray(source.gps_time)[came_from]
    )

    # Coordinates are not reprojected here, only reordered: the stored
    # integers should be identical, not merely close.
    np.testing.assert_array_equal(
        sorted_cloud.X, np.asarray(source.X)[came_from]
    )
    np.testing.assert_array_equal(
        sorted_cloud.Y, np.asarray(source.Y)[came_from]
    )
    np.testing.assert_array_equal(
        sorted_cloud.Z, np.asarray(source.Z)[came_from]
    )


def test_orig_index_rows_are_unique(tmp_path):
    """No point may be written twice or lost by the run-coalescing gather."""
    _, out_las, _, counts = run_sort(tmp_path)
    came_from = np.asarray(laspy.read(out_las).orig_index)
    assert len(np.unique(came_from)) == counts["kept"]


def test_chunk_size_does_not_change_the_result(tmp_path):
    """One chunk or many, the sorted file must come out identical."""
    las_path, centerline_path = build_corridor_cloud(tmp_path)

    outputs = []
    for chunk_size in (10_000, 700, 64):
        out_las = str(tmp_path / f"sorted{chunk_size}.las")
        sort_las_by_station(
            las_path,
            centerline_path,
            40.0,
            out_las,
            str(tmp_path / f"index{chunk_size}.csv"),
            chunk_size=chunk_size,
        )
        outputs.append(laspy.read(out_las))

    for other in outputs[1:]:
        np.testing.assert_array_equal(
            np.asarray(outputs[0].orig_index), np.asarray(other.orig_index)
        )
        np.testing.assert_array_equal(
            np.asarray(outputs[0].station), np.asarray(other.station)
        )


# --------------------------------------------------------------------------
# 3. the corridor cull
# --------------------------------------------------------------------------

def test_corridor_cull_drops_wide_points(tmp_path):
    road_width = 40.0
    _, out_las, _, counts = run_sort(tmp_path, road_width=road_width)

    offset = np.asarray(laspy.read(out_las).offset)
    assert np.all(np.abs(offset) <= road_width / 2.0)

    assert counts["dropped_by_cull"] > 0, "test cloud should be wider than the road"
    assert (
        counts["processed"]
        == counts["kept"] + counts["dropped_by_cull"] + counts["skipped_invalid"]
    )


def test_a_wider_road_keeps_more_points(tmp_path):
    # Two full runs, each in its own directory so the outputs cannot collide.
    narrow_dir, wide_dir = tmp_path / "narrow", tmp_path / "wide"
    narrow_dir.mkdir()
    wide_dir.mkdir()

    _, _, _, narrow = run_sort(narrow_dir, road_width=20.0)
    _, _, _, wide = run_sort(wide_dir, road_width=80.0)
    assert wide["kept"] > narrow["kept"]
    assert wide["processed"] == narrow["processed"]


# --------------------------------------------------------------------------
# 4. build_foot_index on its own
# --------------------------------------------------------------------------

def test_empty_feet_are_not_special_cased(tmp_path):
    """A foot with no points gets the next occupied row, so its range is empty."""
    # Nothing between 2 and 5.
    stations = np.array([0.2, 0.7, 1.5, 5.1, 5.9, 6.0])
    feet, starts = build_foot_index(stations, 1.0)

    np.testing.assert_array_equal(feet, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(starts, [0, 2, 3, 3, 3, 3, 5, 6])

    # Feet 2, 3 and 4 are empty: start == next start.
    for i in (2, 3, 4):
        assert starts[i] == starts[i + 1]


def test_foot_index_starts_at_or_below_the_first_station():
    stations = np.array([12.4, 12.9, 13.1])
    feet, starts = build_foot_index(stations, 1.0)
    assert feet[0] == 12.0
    assert starts[0] == 0
    assert feet[-1] == 14.0 and starts[-1] == 3      # the sentinel


def test_foot_step_controls_granularity():
    stations = np.array([0.0, 0.4, 0.9, 1.4])
    feet, starts = build_foot_index(stations, 0.5)
    np.testing.assert_allclose(feet, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(starts, [0, 2, 3, 4])


def test_foot_index_of_nothing_is_empty():
    feet, starts = build_foot_index(np.empty(0), 1.0)
    assert len(feet) == 0 and len(starts) == 0
