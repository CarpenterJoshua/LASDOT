"""
Reorganise a roadway point cloud into STATION ORDER, and index it by the foot.

Two files come out:

    a sorted LAS   the same cloud, same attributes, with its points rewritten
                   in order of increasing station instead of scan order.
    an index CSV   one row per even-foot station: `station_ft,start_index`,
                   where start_index is the 0-based row in the sorted LAS at
                   which that foot begins.

Together they are what makes a cross-section cheap. To pull the window
[lo, hi] out of a billion-point cloud you look up two numbers in the CSV --
the start of foot `lo` and the start of foot `hi + 1` -- and read exactly that
row range from the sorted LAS. One seek, one contiguous read, no scanning.

    start = start_index of foot lo
    stop  = start_index of the next foot past hi     (the CSV's own next row)
    reader.seek(start); points = reader.read_points(stop - start)

Points further from the centerline than `road_width / 2` are dropped on the
way through, so the output is the corridor only.

MEMORY. The cloud itself is never held whole: it is read in chunks and the
point data is thrown away as soon as each chunk has been projected. What IS
held whole is three light arrays over the *surviving* points -- station,
offset, and source row -- at 24 bytes per survivor. That is ~2.4 GB at 100
million survivors, which is the practical ceiling of this script.

SPEED. The second pass reads source points in station order, which is
scattered with respect to a scan-ordered file. `gather_points_by_source_row`
below coalesces consecutive rows into runs to keep that cheap; it is a
one-time cost, and it is much heavier on `.laz` (every seek decompresses a
chunk) than on plain `.las`.

`sort_las_by_station` is the whole job in one call; everything above it is a
step of that call, split out so each step can be read and tested on its own.
"""

from __future__ import annotations

from copy import deepcopy

import laspy
import numpy as np

from station_projection import load_centerline, prepare_centerline, xyz2sta


# --------------------------------------------------------------------------
# Layer 1: the pure core -- arrays in, arrays out, no files
# --------------------------------------------------------------------------

def concatenate_and_free(pieces, dtype):
    """
    Join a list of per-chunk arrays into one array, releasing each piece as
    soon as it has been copied.

    `np.concatenate(pieces)` would do the same job in one line, but it holds
    every piece AND the finished array at the same time -- a transient 2x
    spike, which on a cloud with a hundred million survivors is gigabytes we
    do not have to spend. Popping as we go means the old pieces are freed
    while the new array fills up.

    NOTE: this empties `pieces`. That is the point.
    """
    total = sum(len(piece) for piece in pieces)
    joined = np.empty(total, dtype=dtype)

    # Reversed, so that pop() -- which takes from the END of a list, and is the
    # only cheap way to shrink one -- hands them back in their original order.
    pieces.reverse()
    at = 0
    while pieces:
        piece = pieces.pop()
        joined[at:at + len(piece)] = piece
        at += len(piece)
    return joined


def build_foot_index(sorted_stations, foot_step):
    """
    Find where each even-foot station begins in an already-sorted station list.

    sorted_stations : (N,) stations in increasing order -- the whole point of
                      the sort above.
    foot_step       : index granularity, in feet. 1.0 means one row per foot.

    Returns (foot_stations, start_indices), both 1-D and the same length.

    The lookup is `np.searchsorted`, a binary search over the sorted stations.
    For a given foot it answers "what is the first row whose station has
    reached this foot?" -- side="left" so a point sitting exactly on the foot
    belongs to that foot, not the one before.

    Feet with no points need no special case. searchsorted just returns the
    row where the NEXT occupied foot begins, so that foot's start equals the
    following foot's start and its range comes out empty -- which is correct.

    The last row is a SENTINEL: one step past the last real foot, pointing at
    the end of the data. It exists so that every foot in the table has a row
    after it, which is what lets a reader compute `stop = next foot's start`
    without checking whether it has run off the end.
    """
    if len(sorted_stations) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=np.int64)

    # Start at the foot boundary at or below the first station, so the very
    # first foot in the table always begins at row 0.
    first_foot = np.floor(sorted_stations[0] / foot_step) * foot_step
    last_station = sorted_stations[-1]

    # How many real feet we need to cover the data, plus one for the sentinel.
    real_feet = int(np.floor((last_station - first_foot) / foot_step)) + 1

    # Built by multiplication, not by adding foot_step over and over: repeated
    # addition of a float accumulates rounding error over thousands of steps.
    foot_stations = first_foot + foot_step * np.arange(real_feet + 1)

    start_indices = np.searchsorted(sorted_stations, foot_stations, side="left")

    # The sentinel is one step past the end of the data, so searchsorted has
    # already returned N for it. Set it outright anyway -- it says plainly that
    # the last row is a stop marker, not a foot with points in it.
    start_indices[-1] = len(sorted_stations)

    return foot_stations, start_indices


# --------------------------------------------------------------------------
# Layer 2: reading -- the streaming projection pass
# --------------------------------------------------------------------------

def scan_for_survivors(source_path, prepared, road_width, chunk_size, k):
    """
    Read the whole cloud once and keep only what the sort needs.

    For every chunk of points we project to station/offset, drop everything
    outside the corridor, and then THROW THE POINTS AWAY, keeping just three
    numbers per survivor:

        station     what we are going to sort by
        offset      carried through so it can be stamped on the output
        source_row  which row of the source file this point came from

    That last one is the trick. It means the second pass can go back and fetch
    the actual point data in any order it likes, and this pass never has to
    hold more than one chunk of real points at a time.

    Returns (stations, offsets, source_rows, counts).
    """
    half_width = road_width / 2.0

    # One entry per chunk; joined at the end by concatenate_and_free.
    station_pieces, offset_pieces, row_pieces = [], [], []
    counts = {
        "processed": 0,
        "kept": 0,
        "dropped_by_cull": 0,
        "skipped_invalid": 0,
    }

    with laspy.open(source_path) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            # chunk.x and friends are lazy views that decode through the
            # header's scale and offset; np.asarray materialises real floats.
            xyz = np.column_stack(
                [
                    np.asarray(chunk.x),
                    np.asarray(chunk.y),
                    np.asarray(chunk.z),
                ]
            )

            # Row numbers in the SOURCE file. Stamped before any culling, so a
            # survivor keeps the row it actually came from.
            source_row = counts["processed"] + np.arange(
                len(chunk), dtype=np.uint64
            )
            counts["processed"] += len(chunk)

            projected = xyz2sta(xyz, prepared, k=k)
            station = projected[:, 0]
            offset = projected[:, 1]

            # A point with a broken coordinate comes back as NaN and cannot be
            # sorted or binned, so it goes.
            usable = np.isfinite(station) & np.isfinite(offset)

            # THE CORRIDOR CULL. Offset is signed -- positive right of travel,
            # negative left -- so the corridor is the band |offset| <= half the
            # road width. Comparing the absolute value keeps both sides in one
            # test. (NaN fails this comparison too, which is harmless: those
            # rows were already excluded by `usable`.)
            inside = np.abs(offset) <= half_width

            keep = usable & inside
            counts["skipped_invalid"] += int((~usable).sum())
            counts["dropped_by_cull"] += int((usable & ~inside).sum())
            counts["kept"] += int(keep.sum())
            if not keep.any():
                continue

            station_pieces.append(station[keep])
            offset_pieces.append(offset[keep])
            row_pieces.append(source_row[keep])

    stations = concatenate_and_free(station_pieces, np.float64)
    offsets = concatenate_and_free(offset_pieces, np.float64)
    source_rows = concatenate_and_free(row_pieces, np.uint64)
    return stations, offsets, source_rows, counts


def gather_points_by_source_row(reader, wanted_rows, source_header):
    """
    Fetch a scattered set of source rows and hand them back IN THE ORDER ASKED.

    reader        : an open laspy reader on the source file.
    wanted_rows   : (N,) source row numbers, in the order we want them out.
    source_header : the source file's header, used to shape the record.

    The rows we want are all over the file, because station order is not scan
    order. Asking for them one at a time would mean N seeks and N one-point
    reads. Instead:

      1. argsort the wanted rows so the file is visited front-to-back --
         disks, and laspy's readers, strongly prefer moving forwards.
      2. Cut that ascending list wherever the row numbers stop being
         consecutive. Each uncut stretch is a RUN of neighbouring rows that
         can be read with a single seek and a single multi-point read.
      3. Scatter each run into its final resting place. `visit_order` still
         remembers where each row belonged in the caller's ordering, so
         `destination[visit_order[a:b]] = run` puts them back.

    Runs are why this is affordable: mobile LiDAR is collected while driving,
    so scan order already roughly follows station, and most of the file comes
    back as long sequential stretches rather than millions of lone points.
    """
    destination = laspy.ScaleAwarePointRecord.zeros(
        len(wanted_rows), header=source_header
    )
    if len(wanted_rows) == 0:
        return destination

    visit_order = np.argsort(wanted_rows, kind="stable")
    ascending = wanted_rows[visit_order]

    # A run breaks wherever the step from one row to the next is not exactly 1.
    # np.diff gives those steps; nonzero(...)[0] gives the positions before a
    # break, and +1 turns them into the positions where the next run starts.
    breaks = np.nonzero(np.diff(ascending) != 1)[0] + 1
    run_starts = np.concatenate([[0], breaks])
    run_stops = np.concatenate([breaks, [len(ascending)]])

    for start, stop in zip(run_starts, run_stops):
        reader.seek(int(ascending[start]))
        run = reader.read_points(int(stop - start))
        # Raw structured-array assignment: same dtype in and out, so the packed
        # bytes move across untouched.
        destination.array[visit_order[start:stop]] = run.array

    return destination


# --------------------------------------------------------------------------
# Layer 3: writing -- the sorted LAS and the index CSV
# --------------------------------------------------------------------------

def write_foot_index_csv(path, foot_stations, start_indices):
    """
    Write the even-foot index.

    Two columns, `station_ft,start_index`. start_index is 0-based and counts
    rows of the SORTED LAS, so it can be handed straight to LasReader.seek.
    The final row is the sentinel described in build_foot_index.
    """
    with open(path, "w") as handle:
        handle.write("station_ft,start_index\n")
        for foot, start in zip(foot_stations, start_indices):
            handle.write(f"{foot:.4f},{int(start)}\n")


def sorted_output_header(source_header):
    """
    Build the output header: the source header, plus three extra dimensions.

    Everything else is copied as-is -- point format, scales, offsets, and the
    projection VLRs. The coordinates are NOT being changed here, only the
    order of the rows, so the file stays in the source CRS and keeps every
    attribute it had.

    The extras:
        orig_index  which row this point had in the source file
        station     its distance along the centerline
        offset      its signed distance from the centerline, + right of travel

    Storing station and offset means a reader never has to re-run the
    projection to know where a point sits.

    (The stale point count and min/max copied along with the header do not
    matter: laspy's writer resets them and recomputes as it writes.)
    """
    header = deepcopy(source_header)
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="orig_index",
            type=np.uint64,
            description="row index in the source file",
        )
    )
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="station",
            type=np.float64,
            description="station along centerline",
        )
    )
    header.add_extra_dim(
        laspy.ExtraBytesParams(
            name="offset",
            type=np.float64,
            description="offset, + right of travel",
        )
    )
    return header


def build_sorted_record(gathered, header, station, offset, orig_index):
    """
    Copy gathered points into a record matching the OUTPUT point format.

    The copy is unavoidable: adding extra dimensions changes the point format,
    and laspy refuses to write a record whose format differs from the writer's
    ("Incompatible point formats"). Copying field by field off `.array` moves
    the packed bytes across wholesale, so the bit-packed dimensions -- return
    numbers, classification flags -- and the stored XYZ integers are not
    reinterpreted on the way. Nothing is requantised; this is a reorder, not a
    conversion.
    """
    out = laspy.ScaleAwarePointRecord.zeros(len(gathered), header=header)

    # The output dtype is the input dtype plus our three extras, so every
    # source field has a home waiting for it.
    for name in gathered.array.dtype.names:
        out.array[name] = gathered.array[name]

    out.orig_index = orig_index
    out.station = station
    out.offset = offset
    return out


def write_sorted_las(source_path, out_path, order, source_rows, stations, offsets, chunk_size):
    """
    Write the reordered cloud.

    `order` is the permutation the argsort produced: order[R] says which entry
    of the survivor arrays belongs in output row R. So walking `order`
    from front to back and appending as we go puts output row R exactly where
    the index CSV says it is.
    """
    with laspy.open(source_path) as reader:
        header = sorted_output_header(reader.header)

        with laspy.open(out_path, mode="w", header=header) as writer:
            for start in range(0, len(order), chunk_size):
                stop = min(start + chunk_size, len(order))

                # Which survivors go in this block of output rows...
                slots = order[start:stop]
                # ...and which source rows those survivors came from.
                wanted_rows = source_rows[slots]

                gathered = gather_points_by_source_row(
                    reader, wanted_rows, reader.header
                )
                out = build_sorted_record(
                    gathered,
                    header,
                    stations[slots],
                    offsets[slots],
                    wanted_rows,
                )
                writer.write_points(out)


# --------------------------------------------------------------------------
# Layer 4: the whole job, and the command line
# --------------------------------------------------------------------------

def sort_las_by_station(source_las, centerline_path, road_width, out_las, out_csv,
    foot_step=1.0, chunk_size=2_000_000, k=8, sample_spacing=1.0, start_station=0.0):
    """
    Sort a cloud by station and write the even-foot index beside it.

    source_las      : the cloud to reorganise.
    centerline_path : plain 2D vertex list, same CRS and units as the cloud.
    road_width      : corridor width in feet. Points more than half this far
                      from the centerline are dropped.
    out_las         : where to write the reordered cloud.
    out_csv         : where to write the even-foot index.
    foot_step       : index granularity in feet; 1.0 gives one row per foot.
    chunk_size      : points held in memory per pass.
    k, sample_spacing, start_station : passed through to the projection.

    Returns the counts dict from the streaming pass, with the index size added.

    The station values and the cull decision are computed ONCE, in
    scan_for_survivors, and both outputs are built from that single result.
    There is no second projection pass that could disagree, so the CSV's row
    numbers cannot drift out of step with the sorted LAS.
    """
    prepared = prepare_centerline(
        load_centerline(centerline_path),
        sample_spacing=sample_spacing,
        start_station=start_station,
    )

    # Pass 1: project everything, keep the corridor, remember only station,
    # offset, and where each survivor came from.
    stations, offsets, source_rows, counts = scan_for_survivors(
        source_las, prepared, road_width, chunk_size, k
    )
    if len(stations) == 0:
        raise ValueError(
            "no points survived the corridor cull -- check that the centerline "
            "is in the same coordinate system as the cloud, and that "
            f"road_width ({road_width}) is wide enough"
        )

    # THE SORT. `argsort` does not sort the values -- it returns POSITIONS, a
    # list of indices such that stations[order[0]] is the smallest station,
    # stations[order[1]] the next, and so on. Positions are what we want,
    # because the offsets and the source row numbers have to be carried along
    # in the same order, and indexing all three arrays by `order` does that.
    #
    # kind="stable" makes ties deterministic: points sharing a station stay in
    # the order they came out of the source file, so two runs on the same
    # input produce byte-identical output.
    order = np.argsort(stations, kind="stable")
    sorted_stations = stations[order]

    # The index, straight off the sorted stations.
    foot_stations, start_indices = build_foot_index(sorted_stations, foot_step)
    write_foot_index_csv(out_csv, foot_stations, start_indices)

    # Pass 2: fetch the real points in that order and write them out.
    write_sorted_las(
        source_las, out_las, order, source_rows, stations, offsets, chunk_size
    )

    counts["index_rows"] = len(foot_stations)
    counts["first_station"] = float(sorted_stations[0])
    counts["last_station"] = float(sorted_stations[-1])
    return counts
