"""
Station/offset projection for roadway point clouds.

This module is the single home for moving between the two coordinate systems a
roadway cloud lives in:

    global   (easting, northing, elevation)  -- how the cloud was surveyed
    station  (station, offset, elevation)    -- how a road is described

where, measured against a 2D centerline (a list of vertices, no curves):

    station = how far along the centerline the point sits
    offset  = how far to the side of the centerline it sits
              (positive = RIGHT of travel, negative = LEFT)

Elevation is never touched by either direction.

Two directions live here:

    FORWARD   global -> station.  Implemented (see `xyz2sta`).
    BACKWARD  station -> global.  Not yet written; the section is marked out
                                  below so it slots in beside the forward code.

The file is built in layers, cheapest and purest first:

    _project_onto_segments   the projection math, and nothing else.
    prepare_centerline       a KD-tree broadphase, so we never test every
                             point against every segment.
    xyz2sta                  forward pure core -- numpy in, numpy out.
    (backward pure core)     to come.

There is no file handling here at all: every function takes arrays and returns
arrays. Callers that have a whole cloud to get through -- sort_by_station.py,
say -- read it in chunks and hand this module one chunk at a time, which is
what keeps a billion-point job on an ordinary workstation.
"""

import io

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# Layer 1: the projection math
# --------------------------------------------------------------------------

def _project_onto_segments(xy, seg_start, direction, length):
    """
    Project each point onto ONE segment. Every array is per-point and already
    gathered, so row i of `xy` is tested against row i of `seg_start`.

    xy         : (N, 2) point positions.
    seg_start  : (N, 2) the "A" vertex of each point's segment.
    direction  : (N, 2) unit vector along that segment.
    length     : (N,)   length of that segment.

    Returns (t, offset, distance):
        t         how far along the segment the point lands, in [0, length]
        offset    signed side-distance, positive RIGHT of travel
        distance  unsigned distance from the point to the segment
    """
    w = xy - seg_start        # (N, 2) vector from the segment start to the point

    # How far along the segment each point lands (its shadow on the line),
    # found by projecting w onto the direction vector.
    t = w[:, 0] * direction[:, 0] + w[:, 1] * direction[:, 1]

    # THE CLAMP. A point whose shadow falls past either end of the segment is
    # folded onto the nearer endpoint. This one line is what makes corners work
    # with no special-case code: at a vertex, both adjacent segments clamp to
    # that shared vertex, and the nearest-foot rule picks between them.
    t = np.clip(t, 0.0, length)

    # The foot of the projection: the actual nearest point on this segment.
    foot = seg_start + t[:, None] * direction
    delta = xy - foot
    distance = np.hypot(delta[:, 0], delta[:, 1])

    # THE SIGN. The 2D cross product of the travel direction with w is positive
    # on the LEFT (in normal easting/northing, where north is "up"). We want
    # offset positive on the RIGHT, so we flip it. A point exactly on the line
    # counts as the right side.
    cross = direction[:, 0] * w[:, 1] - direction[:, 1] * w[:, 0]
    side = np.where(cross > 0.0, -1.0, 1.0)

    return t, side * distance, distance


# --------------------------------------------------------------------------
# Layer 2: the centerline and its broadphase index
# --------------------------------------------------------------------------

class PreparedCenterline:
    """A centerline with a KD-tree built over it. Build once, reuse forever."""

    def __init__(
        self,
        seg_start,       # (S, 2) each segment's start vertex
        direction,       # (S, 2) unit vector along each segment
        length,          # (S,)   each segment's length
        station0,        # (S,)   station at each segment's start
        sample_segment,  # (P,)   which segment owns each tree sample
        tree,            #        the samples themselves
        total_length,    #        summed length of every segment
        start_station,   #        station value at the first vertex
    ):
        self.seg_start = seg_start
        self.direction = direction
        self.length = length
        self.station0 = station0
        self.sample_segment = sample_segment
        self.tree = tree
        self.total_length = total_length
        self.start_station = start_station

    def __repr__(self):
        return (
            f"{type(self).__name__}(segments={len(self.length)}, "
            f"samples={self.tree.n}, "
            f"station={self.start_station:g}.."
            f"{self.start_station + self.total_length:g})"
        )


def load_centerline(path):
    """
    Read centerline vertices from a text file: two numbers per line
    (easting, northing), comma-, space-, or tab-separated, in the SAME
    coordinate system and linear units as the point cloud.

    Extra columns (a 3D export, say) are ignored.
    """
    with open(path) as handle:
        text = handle.read().replace(",", " ")
    vertices = np.loadtxt(io.StringIO(text), ndmin=2)
    return vertices[:, :2]


def prepare_centerline(vertices, sample_spacing=1.0, start_station=0.0):
    """
    Turn a list of vertices into a PreparedCenterline.

    vertices       : (M, 2) ordered vertices, running from start to end in the
                     direction of increasing station.
    sample_spacing : how finely to densify the centerline for the KD-tree, in
                     coordinate units. Smaller is more robust and uses more
                     memory; it only ever holds len(centerline)/spacing points.
    start_station  : station value of the first vertex. Defaults to 0, i.e.
                     station is pure geometric arc length.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 2:
        raise ValueError("centerline needs at least 2 vertices")
    vertices = vertices[:, :2]

    span = vertices[1:] - vertices[:-1]
    length = np.hypot(span[:, 0], span[:, 1])

    # Drop repeated vertices; a zero-length segment has no direction and would
    # divide by zero below.
    usable = length > 0.0
    if not usable.any():
        raise ValueError("centerline has no segments with non-zero length")
    seg_start = vertices[:-1][usable]
    length = length[usable]
    direction = span[usable] / length[:, None]

    # Station at the START of each segment: the running sum of the lengths
    # before it.
    station0 = start_station + np.concatenate([[0.0], np.cumsum(length)[:-1]])

    # THE SPRINKLE. This loop is where the broadphase index is built, and it
    # exists to work around one fact: a KD-tree can only index POINTS, but what
    # we actually need to search is SEGMENTS. So we scatter points along every
    # segment and remember which segment each one came from -- `samples` holds
    # the scattered points, `owners` the lookup back to their segment. Together
    # they turn "which segment is nearest?" into "which sample is nearest?",
    # which is a question a tree can answer.
    #
    # Nothing about station or offset is decided here. This only ever produces
    # a good SHORTLIST of segments to test properly later; see xyz2sta.
    #
    # Each segment contributes evenly spaced samples INCLUDING both endpoints,
    # so even a segment shorter than sample_spacing is represented and corners
    # always have a sample sitting exactly on them. That last part matters:
    # the clamp in _project_onto_segments folds both segments meeting at a
    # corner onto that shared vertex, so both must reach the shortlist for the
    # tie-break to choose between them.
    samples, owners = [], []
    for i, seg_length in enumerate(length):
        count = max(int(np.ceil(seg_length / sample_spacing)), 1)
        t = np.linspace(0.0, seg_length, count + 1)
        samples.append(seg_start[i] + t[:, None] * direction[i])
        # int32 keeps the per-chunk candidate array half the size it would
        # otherwise be; no centerline has 2 billion segments.
        owners.append(np.full(t.size, i, dtype=np.int32))

    return PreparedCenterline(
        seg_start=seg_start,
        direction=direction,
        length=length,
        station0=station0,
        sample_segment=np.concatenate(owners),
        tree=cKDTree(np.vstack(samples)),
        total_length=float(length.sum()),
        start_station=float(start_station),
    )


# --------------------------------------------------------------------------
# Layer 3: FORWARD projection -- the pure core (global -> station)
# --------------------------------------------------------------------------

def xyz2sta(points_xyz, centerline, k=8, start_station=0.0):
    """
    Project points onto a centerline.

    points_xyz : (N, 3) array of [easting, northing, elevation].
    centerline : either raw (M, 2) vertices or a PreparedCenterline. Pass a
                 prepared one when projecting many chunks against the same
                 centerline -- building the index is the expensive part.
    k          : how many nearest tree samples to turn into candidate segments.
    start_station : only used when `centerline` is raw vertices; for a prepared
                 centerline it is already baked in.

    Returns    : (N, 3) array of [station, offset, elevation], one row per
                 input point, in the same order. Rows whose easting/northing
                 was not finite come back as NaN station and offset.

    Each point is matched to its single nearest point on the centerline (the
    "nearest-foot" rule). Points far from the road still get values; if you
    only want the corridor, filter afterwards on offset.

    Two behaviours worth knowing:
      * On an exact distance tie -- the seam on the inside of a bend -- the
        LOWER segment index wins, so results are deterministic.
      * Station is not monotonic across such a seam. That is inherent to
        stationing against a polyline, not a bug here.
    """
    points_xyz = np.asarray(points_xyz, dtype=float)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError("points_xyz must be an (N, 3) array")

    if isinstance(centerline, PreparedCenterline):
        if start_station:
            raise ValueError(
                "start_station is baked into a PreparedCenterline; "
                "pass it to prepare_centerline() instead"
            )
        prepared = centerline
    else:
        prepared = prepare_centerline(centerline, start_station=start_station)

    xy = points_xyz[:, :2]
    z = points_xyz[:, 2]

    station = np.full(len(xy), np.nan)
    offset = np.full(len(xy), np.nan)

    # A non-finite coordinate would make the KD-tree query raise, so those rows
    # sit the query out and stay NaN.
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return np.column_stack([station, offset, z])
    query_xy = xy[finite]

    # THE BROADPHASE. Here is the KD-tree's entire job, and it is only ever to
    # produce a SHORTLIST. The tree holds the sprinkled samples built by
    # prepare_centerline, so what comes back is the k nearest SAMPLES;
    # sample_segment then translates those into the k candidate SEGMENTS.
    #
    # No station or offset exists yet. The tree's own distances are distances
    # to sample points rather than to segments, so they are approximate and get
    # dropped on the floor (`_`) the moment they have done their job of ranking.
    #
    # Why k=8 and not 1: nearest-sample and nearest-segment are not always the
    # same segment, most often at bends where segments crowd together. The tree
    # only has to land the right segment somewhere in its top k -- the exact
    # math below does the actual choosing.
    k_eff = min(k, prepared.tree.n)
    _, sample_idx = prepared.tree.query(query_xy, k=k_eff)
    candidates = prepared.sample_segment[
        sample_idx.reshape(len(query_xy), k_eff)
    ]                                                  # (Nq, k) segment ids
    del sample_idx                                     # (Nq, k) of int64

    best_distance = np.full(len(query_xy), np.inf)
    best_station = np.zeros(len(query_xy))
    best_offset = np.zeros(len(query_xy))
    best_segment = np.full(len(query_xy), np.iinfo(np.int32).max, dtype=np.int32)

    # THE NARROWPHASE. The true station and offset are computed HERE, and
    # nowhere else in this function. Each candidate segment gets the exact
    # projection math from layer 1, and the genuinely closest one wins. The
    # tree's ranking carries no authority at this point: a segment that came
    # back 8th can win outright, it is just that the 1st usually does.
    #
    # A small fixed loop over the k candidate slots, keeping a running best.
    # `candidates` is unavoidably (Nq, k), but doing the *math* one slot at a
    # time keeps every intermediate (Nq,). Vectorising across k as well would
    # make each of the half-dozen temporaries below (Nq, k) too -- at 5M points
    # and k=8 that is ~320 MB apiece.
    for j in range(k_eff):
        segment = candidates[:, j]
        t, offset_j, distance_j = _project_onto_segments(
            query_xy,
            prepared.seg_start[segment],
            prepared.direction[segment],
            prepared.length[segment],
        )
        # The real station for this candidate: the station already accumulated
        # at the segment's start, plus how far along that segment the point
        # landed. `offset_j` and `distance_j` come straight out of the same
        # call. These three lines are the payload the whole file exists for.
        station_j = prepared.station0[segment] + t

        # Closer wins; on an exact tie the lower segment index wins.
        better = (distance_j < best_distance) | (
            (distance_j == best_distance) & (segment < best_segment)
        )
        best_distance = np.where(better, distance_j, best_distance)
        best_station = np.where(better, station_j, best_station)
        best_offset = np.where(better, offset_j, best_offset)
        best_segment = np.where(better, segment, best_segment)

    station[finite] = best_station
    offset[finite] = best_offset
    return np.column_stack([station, offset, z])


# --------------------------------------------------------------------------
# Layer 4: BACKWARD projection -- the pure core (station -> global)
# --------------------------------------------------------------------------
#
# NOTHING HERE YET. This section is reserved so backward projection lands
# beside its forward twin instead of in a file of its own.
#
# The intended entry point mirrors xyz2sta:
#
#     sta2xyz(station_offset_z, centerline, ...)
#         station_offset_z : (N, 3) array of [station, offset, elevation]
#         centerline       : raw vertices or a PreparedCenterline, same as above
#         returns          : (N, 3) array of [easting, northing, elevation]
#
# It needs no KD-tree. The work is: find which segment owns each station with
# np.searchsorted on prepared.station0, walk (station - station0) along that
# segment's `direction` from its `seg_start`, then step `offset` along the
# right-hand normal of that direction -- which is `direction` rotated -90
# degrees, i.e. (direction_y, -direction_x), matching the sign convention set
# in _project_onto_segments.
#
# Round-tripping is exact only for points inside the corridor and away from the
# seams at bends; outside those, forward projection is many-to-one (the clamp)
# and cannot be undone. Say so in the docstring when it is written.
#
# --------------------------------------------------------------------------
