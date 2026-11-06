"""Task 2: nearest neighbour search on the EV Charging Station dataset.

For each of the 200 query points, find the nearest facility with three methods:
  1. Sequential Scan: measure the distance from the query to every point.
  2. Best-First search (BF) on an R-tree built with the group's shared module.
  3. BF with Divide-and-Conquer: sort the points by y, cut them into four equal groups,
     build one R-tree per group, run BF on all four and keep the closest result.

Run it from the folder that holds this file, rtree_shared.py and the Task2_Datasets folder:
    python task2_nn.py
Optional arguments: the node capacity B (default 4) and the number of decimals printed
(default 2), for example:
    python task2_nn.py 8 3
The results are printed on screen and saved to task2_output.txt.
Imports: time, math, heapq, sys, and the shared R-tree module rtree_shared, which must sit
beside this file.
"""
import time      # perf_counter, the timer used for the running times
import math      # sqrt, used once per query to turn a squared distance into a distance
import heapq     # heappush and heappop, the priority queue used by Best-First search
import sys       # argv (optional command-line settings) and modules (which rtree_shared was loaded)

from rtree_shared import RTree      # the group's shared R-tree: node, rectangle, insertion and splitting

DATA_FILE = "Task2_Datasets/ev_charging_station.txt"   # the facility points, one 'id x y' per line
QUERY_FILE = "Task2_Datasets/query_points_BF.txt"      # the 200 query points, in the same line format
OUTPUT_FILE = "task2_output.txt"                # where the results are saved
B = 4                                           # default maximum number of entries in an R-tree node
DECIMALS = 2                                    # default number of decimals printed


def read_points(path):
    """Read a file of 'id x y' lines and return a list of (id, x, y) tuples.
    A line that is not three numbers raises an error and stops the program."""
    points = []  # the list of (id, x, y) tuples that we return
    with open(path) as f:                       # the file closes by itself after this block
        for line in f:                          # one line per point
            parts = line.split()                # cut the line into its three words: id, x, y
            if not parts:                       # skip blank lines
                continue
            # The id is a whole number; x and y can have decimals.
            points.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return points


def sequential_scan(points, queries):
    """Method 1. For each query, measure the distance to every point and keep the closest.

    The smallest squared distance wins. Squared distance ranks points in the same order as the
    real distance and saves a square root per point. If two points are equally close, the one
    with the smaller id wins, so the comparison key is (squared distance, id).
    """
    results = []  # one result row per query
    for qid, qx, qy in queries:                 # one search per query point
        best_key = None  # (squared distance, id) of the closest point so far; None until the first point
        best_point = None  # the closest point so far as (id, x, y)
        for pid, px, py in points:              # test every point in the dataset
            dx = px - qx                        # horizontal gap between the point and the query
            dy = py - qy                        # vertical gap between the point and the query
            key = (dx * dx + dy * dy, pid)      # squared distance (Pythagoras), id as tie-breaker
            if best_key is None or key < best_key:      # the first point, or a closer one
                best_key = key  # this point is now the closest, so remember its key
                best_point = (pid, px, py)      # remember this point as the closest so far
        # The square root is taken once per query, only for the distance we print. The search
        # above compared squared distances, which gives the same winner.
        results.append((qid, best_point[0], best_point[1], best_point[2], math.sqrt(best_key[0])))
    return results


def build_rtree(points, capacity):
    """Build one R-tree with the shared module, inserting the points in the order given."""
    tree = RTree(B=capacity)                    # an empty tree whose nodes hold at most `capacity` entries
    for pid, px, py in points:                  # insert the points one at a time
        tree.insert_point({'id': pid, 'x': px, 'y': py})    # the shared module stores a point as a dictionary
    return tree


def mindist_sq(mbr, qx, qy):
    """Squared shortest distance from the query to a rectangle (0 if the query is inside it)."""
    # Gap along x. max picks the case that applies. Query left of the box: x1 - qx is positive.
    # Query inside the box's x range: both differences are negative, so 0 wins. Query right of
    # the box: qx - x2 is positive.
    dx = max(mbr['x1'] - qx, 0.0, qx - mbr['x2'])
    # Gap along y, with the same three cases: query below the box, inside its y range, or above it.
    dy = max(mbr['y1'] - qy, 0.0, qy - mbr['y2'])
    return dx * dx + dy * dy                    # Pythagoras on the two gaps


def bf_nearest(tree, qx, qy):
    """Find the point nearest to (qx, qy) using Best-First search on one R-tree.

    Returns ((squared distance, id), point). The search keeps a priority queue (a heap) and always
    handles the entry with the smallest distance key first. A node's key is the shortest possible
    distance from the query to its rectangle, so no point inside it can be closer than that (a
    lower bound). A point's key is its own distance. Squared distances are used throughout: they
    rank entries in the same order as real distances and save a square root per entry.
    """
    # Queue entry: (squared distance, counter, is_point, item). The counter is unique and goes up
    # by 1 on every push, so entries with equal distances leave in the order they were added.
    # is_point says whether the item is a point or a node. It is never compared, because the
    # counter before it is always different.
    counter = 0
    # Start the queue with the root, keyed by the distance from the query to the root's rectangle.
    heap = [(mindist_sq(tree.root.MBR, qx, qy), counter, False, tree.root)]
    best_key = None  # (squared distance, id) of the best point found; None until one is found
    best_point = None  # the best point itself (a dictionary); None until one is found
    while heap:                                 # keep going while entries remain in the queue
        # Stop check. With no best point yet we cannot stop. Once we have one, we stop only when
        # the smallest key left is STRICTLY larger than the best distance. An entry at exactly the
        # best distance could still hold a point with a smaller id, which would win the tie.
        if best_key is not None and heap[0][0] > best_key[0]:
            break
        key, _, is_point, item = heapq.heappop(heap)    # take the entry with the smallest key
        if is_point:
            # A point reached the front of the queue, so it is a candidate answer.
            candidate = (key, item['id'])       # rank by distance first, then by the smaller id
            if best_key is None or candidate < best_key:    # the first candidate, or a better one
                best_key = candidate            # new best distance and id
                best_point = item               # new best point
        elif item.is_leaf():
            # A leaf holds points: add each one to the queue, keyed by its own squared distance.
            for p in item.data_points:
                dx = p['x'] - qx                # horizontal gap between the point and the query
                dy = p['y'] - qy                # vertical gap between the point and the query
                counter += 1
                heapq.heappush(heap, (dx * dx + dy * dy, counter, True, p))     # True marks a point
        else:
            # An internal node holds child nodes: add each child, keyed by the shortest possible
            # distance from the query to that child's rectangle.
            for child in item.child_nodes:
                counter += 1
                heapq.heappush(heap, (mindist_sq(child.MBR, qx, qy), counter, False, child))    # False marks a node
    # The caller takes the square root once per query, only for printing.
    return best_key, best_point


def best_first_search(tree, queries):
    """Method 2. Run Best-First search on the one R-tree for every query."""
    results = []  # one result row per query
    for qid, qx, qy in queries:                 # one search per query point
        best_key, p = bf_nearest(tree, qx, qy)  # the nearest point and its (squared distance, id)
        # The square root is taken once, only for the printed distance. The search compared
        # squared distances, which gives the same winner.
        results.append((qid, p['id'], p['x'], p['y'], math.sqrt(best_key[0])))
    return results


def partition_by_y(points):
    """Sort the points by y and cut them into 4 groups of nearly equal size.

    The sort order is (y, x, id) ascending, so points with the same y are ordered by x and then
    by id. With n points each group gets n // 4 points, and the first n % 4 groups take one extra.
    """
    ordered = sorted(points, key=lambda p: (p[2], p[1], p[0]))     # p is (id, x, y): sort by y, then x, then id
    size, extra = divmod(len(ordered), 4)       # size = points per group, extra = groups that get one more
    groups = []  # will hold the four groups
    start = 0  # where the next group begins in the sorted list
    for i in range(4):                          # build the four groups in order
        end = start + size + (1 if i < extra else 0)    # the first `extra` groups take one extra point
        groups.append(ordered[start:end])       # this group's slice of the sorted list
        start = end                             # the next group starts where this one ended
    return groups


def best_first_divide_and_conquer(trees, queries):
    """Method 3. For each query, run Best-First search on each group's own R-tree to get four
    candidates, then keep the best one by (squared distance, id). This is the same key that
    Method 1 and Method 2 use, so equal distances are settled the same way."""
    results = []  # one result row per query
    for qid, qx, qy in queries:                 # one search per query point
        best_key = None  # (squared distance, id) of the best candidate across the four trees
        best_point = None  # the best candidate point across the four trees
        for tree in trees:                      # search each of the four group trees
            key, p = bf_nearest(tree, qx, qy)   # this group's candidate
            if best_key is None or key < best_key:      # keep it if it is the first or beats the best so far
                best_key = key  # this candidate is now the best, so remember its key
                best_point = p  # and remember its point
        # The square root is taken once per query, only for the printed distance.
        results.append((qid, best_point['id'], best_point['x'], best_point['y'], math.sqrt(best_key[0])))
    return results


def format_section(title, results, seconds, decimals):
    """Build one output section as text: a title, one line per query and a running-time line.
    Coordinates and distances print with `decimals` decimal places."""
    lines = [title]  # the section starts with its title line
    for qid, pid, px, py, dist in results:      # one output line per query
        # Example of one line: Query 1: id=56, x=34.15, y=149.21, distance=2.37
        lines.append(f"Query {qid}: id={pid}, x={px:.{decimals}f}, y={py:.{decimals}f}, "
                     f"distance={dist:.{decimals}f}")
    lines.append(f"Total running time for {len(results)} queries: {seconds:.6f} seconds")
    return "\n".join(lines)                     # join the lines into one block of text


def main():
    # Node capacity B: default 4; an optional first command-line argument overrides it.
    capacity = int(sys.argv[1]) if len(sys.argv) > 1 else B
    # Decimal places printed for coordinates and distances: default 2 (as in the assignment
    # example); an optional second command-line argument overrides it.
    decimals = int(sys.argv[2]) if len(sys.argv) > 2 else DECIMALS
    points = read_points(DATA_FILE)             # the facility points
    queries = read_points(QUERY_FILE)           # the 200 query points
    print(f"Read {len(points)} data points and {len(queries)} queries")
    print(f"R-tree module in use: {sys.modules['rtree_shared'].__file__}")     # shows which copy of the module was loaded

    # Each running time covers only that method's search over the 200 queries. Reading the files,
    # building the R-trees and writing the output are outside the timers.
    start = time.perf_counter()  # start the timer for the scan
    scan_results = sequential_scan(points, queries)  # run the scan for all 200 queries
    scan_seconds = time.perf_counter() - start  # elapsed seconds for the whole search
    print(f"Sequential Scan: {scan_seconds:.6f} seconds")

    tree = build_rtree(points, capacity)        # one R-tree over all the points (not timed)
    print(f"Built R-tree with B={capacity}")
    start = time.perf_counter()  # start the timer for BF
    bf_results = best_first_search(tree, queries)  # run BF for all 200 queries
    bf_seconds = time.perf_counter() - start  # elapsed seconds for BF
    print(f"BF: {bf_seconds:.6f} seconds")

    groups = partition_by_y(points)             # four groups of points, in ascending y order
    trees = [build_rtree(g, capacity) for g in groups]      # one R-tree per group (not timed)
    print(f"Built 4 group R-trees with B={capacity}, group sizes {[len(g) for g in groups]}")
    start = time.perf_counter()  # start the timer for BF with Divide-and-Conquer
    dc_results = best_first_divide_and_conquer(trees, queries)  # run it for all 200 queries
    dc_seconds = time.perf_counter() - start  # elapsed seconds for BF with Divide-and-Conquer
    print(f"BF with Divide-and-Conquer: {dc_seconds:.6f} seconds")

    # Display and save the results. This happens after all three timers have stopped, so the
    # printing and the file writing are not counted in any running time.
    report = "\n\n".join([
        format_section("Sequential Scan", scan_results, scan_seconds, decimals),
        format_section("Best-First Search (BF)", bf_results, bf_seconds, decimals),
        format_section("BF with Divide-and-Conquer", dc_results, dc_seconds, decimals),
    ]) + "\n"
    print()
    print(report, end="")                       # display on screen
    with open(OUTPUT_FILE, "w") as out:         # save to the single output file
        out.write(report)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":                      # run main() only when this file is run directly
    main()
