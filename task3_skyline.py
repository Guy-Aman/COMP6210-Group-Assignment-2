"""
Task 3: Skyline Search (BBS)  --  COMP3210/6210 Assignment 2
Smaller is better in both x (total cost) and y (price per m^2).

Run:   python task3_skyline.py [data_file] [B] [output_file]
Default: python task3_skyline.py city3.txt 50 task3_output.txt
Needs rtree_shared.py in the same folder (the shared group R-tree module).
Libraries used: time, sys, heapq (standard library only).
Only the search calls are timed (reading, R-tree construction, writing are not).
"""
import sys
import time
import heapq
from rtree_shared import RTree, load_points


def dominates(p, q):
    """p dominates q: no worse in both dims, strictly better in at least one."""
    return p[0] <= q[0] and p[1] <= q[1] and (p[0] < q[0] or p[1] < q[1])


def order(sky):
    return sorted(sky, key=lambda t: (t[1], t[2], t[0]))      # (id, x, y) by x then y


# ---------------- Method 1: sequential scan ----------------
def sequential_skyline(points):
    """One pass over the data; each point is checked against the current skyline
    window (a point dominated by any dataset point is also dominated by some skyline
    point, so this equals checking every point against all others)."""
    window = []                                  # list of (id, x, y)
    for p in points:
        q = (p['x'], p['y'])
        dominated = False
        keep = []
        for w in window:
            wp = (w[1], w[2])
            if dominates(wp, q):
                dominated = True
                keep = window                    # window unchanged
                break
            if not dominates(q, wp):
                keep.append(w)
        if not dominated:
            keep.append((p['id'], p['x'], p['y']))
            window = keep
    return order(window)


# ---------------- Method 2: BBS on one R-tree ----------------
def bbs(tree):
    """Branch-and-Bound Skyline. Heap key = x1 + y1 (L1 distance of the lower-left
    corner to the origin); counter c breaks ties so Nodes are never compared."""
    sky = []                                     # (id, x, y)
    heap, c = [], 0
    m = tree.root.MBR
    if m['x1'] == float('inf'):                  # empty tree
        return sky
    heapq.heappush(heap, (m['x1'] + m['y1'], c, False, tree.root)); c += 1
    while heap:
        _, _, is_pt, item = heapq.heappop(heap)
        corner = (item['x'], item['y']) if is_pt else (item.MBR['x1'], item.MBR['y1'])
        if any(dominates((s[1], s[2]), corner) for s in sky):
            continue                             # pruned: dominated entry / point
        if is_pt:
            sky.append((item['id'], item['x'], item['y']))
        elif item.is_leaf():
            for p in item.data_points:
                if not any(dominates((s[1], s[2]), (p['x'], p['y'])) for s in sky):
                    heapq.heappush(heap, (p['x'] + p['y'], c, True, p)); c += 1
        else:
            for ch in item.child_nodes:
                k = (ch.MBR['x1'], ch.MBR['y1'])
                if not any(dominates((s[1], s[2]), k) for s in sky):
                    heapq.heappush(heap, (k[0] + k[1], c, False, ch)); c += 1
    return order(sky)


# ---------------- Method 3: BBS + divide and conquer ----------------
def split_points(points):
    """Sort by x (then y, id) and cut into two equal-sized halves."""
    pts = sorted(points, key=lambda p: (p['x'], p['y'], p['id']))
    mid = len(pts) // 2
    return pts[:mid], pts[mid:]


def merge_local(left, right):
    """1-D dominance screening. Every left point has x <= every right point's x, so a
    right point r is dominated by the left skyline iff the best (minimum) left y beats it
    (y strictly smaller, or equal y with a strictly smaller x). Equal-x ties on the
    boundary are handled so a left point can also be removed by a right point."""
    if not left:
        return order(right)
    if not right:
        return order(left)
    min_y = min(l[2] for l in left)
    min_x_at_min_y = min(l[1] for l in left if l[2] == min_y)
    kept_r = [r for r in right
              if r[2] < min_y or (r[2] == min_y and r[1] <= min_x_at_min_y)]
    kept_l = [l for l in left
              if not any(r[1] == l[1] and r[2] < l[2] for r in kept_r)]
    return order(kept_l + kept_r)


def bbs_dc(tree_left, tree_right):
    return merge_local(bbs(tree_left), bbs(tree_right))


def build_tree(points, B):
    t = RTree(B=B)
    for p in points:                             # file order -> identical tree for everyone
        t.insert_point(p)
    return t


def timed(fn, *args):
    t0 = time.perf_counter()
    r = fn(*args)
    return r, time.perf_counter() - t0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "city3.txt"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    out = sys.argv[3] if len(sys.argv) > 3 else "task3_output.txt"

    points = load_points(path)
    print(f"Loaded {len(points)} points from {path}; B={B}")

    t0 = time.perf_counter()
    tree = build_tree(points, B)
    left_pts, right_pts = split_points(points)
    tl, tr = build_tree(left_pts, B), build_tree(right_pts, B)
    print(f"R-trees built in {time.perf_counter() - t0:.2f}s (not timed in results)")

    seq, t_seq = timed(sequential_skyline, points)
    bb, t_bbs = timed(bbs, tree)
    dc, t_dc = timed(bbs_dc, tl, tr)

    same = (seq == bb == dc)
    lines = []
    for name, res, t in (("Sequential Scan", seq, t_seq), ("BBS", bb, t_bbs),
                         ("BBS with Divide-and-Conquer", dc, t_dc)):
        lines.append(f"===== {name} =====")
        lines.append(f"Skyline size: {len(res)}")
        for (i, x, y) in res:
            lines.append(f"id={i}, x={x}, y={y}")
        lines.append(f"Total running time: {t:.6f} seconds")
        lines.append("")
    lines.append(f"Dataset: {path} ({len(points)} points), node capacity B={B}")
    lines.append(f"All three methods return the same skyline: {same}")
    text = "\n".join(lines)
    with open(out, "w") as f:
        f.write(text + "\n")
    # display: full skyline is in the file; print a readable summary too
    print(text)


if __name__ == "__main__":
    main()
