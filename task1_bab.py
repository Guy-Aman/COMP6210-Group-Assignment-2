"""
Task 1 - Nearest neighbour search with Branch-and-Bound.
Runs sequential scan, BaB, and BaB with divide-and-conquer on the Parking data.
Run:    python task1_bab.py [B] (22 taken for the output)     (B = node capacity, default 50)
Output: task1_output.txt
"""

import sys
import math
import time
from rtree_shared import RTree, load_points

PARKING = "Task1_Datasets/parking.txt"
QUERIES = "Task1_Datasets/query_points_BaB.txt"
OUTPUT = "task1_output.txt"
DEFAULT_B = 50

# Helper functions for distance calculation

def dist2(q, p):
    dx = q['x'] - p['x']
    dy = q['y'] - p['y']
    return dx * dx + dy * dy

def minimum_dist(q, mbr):
    """Squared distance from q to the nearest point of an MBR (dict x1,y1,x2,y2)."""
    qx, qy = q['x'], q['y']
    rx = mbr['x1'] if qx < mbr['x1'] else (mbr['x2'] if qx > mbr['x2'] else qx)
    ry = mbr['y1'] if qy < mbr['y1'] else (mbr['y2'] if qy > mbr['y2'] else qy)
    dx = qx - rx
    dy = qy - ry
    return dx * dx + dy * dy
    
def is_better(d2, p, best_d2, best_p):
    """Shared tie rule: smaller distance wins; on an exact tie the smaller id wins."""
    return d2 < best_d2 or (d2 == best_d2 and p['id'] < best_p['id'])


# Program Design - Method 1: Sequential Scan

def sequential_scan(candidate_points, query_point):
    """Check every point and return (nearest point, distance)."""
    
    if not candidate_points:
        return None, float('inf')
    
    nearest_point, min_dist_sq = None, math.inf
    for current_point in candidate_points:
        d2 = dist2(query_point, current_point)
        
        # Update if it's closer, or if it's an equidistant tie with a smaller ID
        if d2 < min_dist_sq or (d2 == min_dist_sq and current_point['id'] < nearest_point['id']):
            nearest_point, min_dist_sq = current_point, d2
            
    return nearest_point, math.sqrt(min_dist_sq)

# Program Design - Method 2: Branch-and-Bound (BaB)

def bab_nn_search(tree, q, trace=False):
    """Branch-and-Bound NN search: depth-first, children visited in ascending mindist,
    a child is pruned when its mindist^2 is larger than the best distance^2 found so far."""
    best = {'p': None, 'd2': math.inf}

    def visit(node, depth):
        pad = "  " * depth
        if node.is_leaf():
            if trace:
                print(f"{pad}Leaf N{node.id}: compute distance to {len(node.data_points)} points")
            for p in node.data_points:
                d2 = dist2(q, p)
                if best['p'] is None or is_better(d2, p, best['d2'], best['p']):
                    best['p'], best['d2'] = p, d2
                    if trace:
                        print(f"{pad}  P{p['id']}({p['x']},{p['y']}) d^2={d2}  -> new best")
                elif trace:
                    print(f"{pad}  P{p['id']}({p['x']},{p['y']}) d^2={d2}")
            return
        ranked = sorted(((minimum_dist(q, c.MBR), c.id, c) for c in node.child_nodes),
                        key=lambda t: (t[0], t[1]))
        if trace:
            print(f"{pad}Node N{node.id}: children by mindist^2 = "
                  f"{[(f'N{c.id}', md) for md, _, c in ranked]}")
        for md, _, child in ranked:
            if md > best['d2']:                     # prune (equal distance is still visited)
                if trace:
                    print(f"{pad}  PRUNE N{child.id}: mindist^2={md} > best d^2={best['d2']}")
                continue
            if trace:
                print(f"{pad}  VISIT N{child.id}: mindist^2={md} <= best d^2={best['d2']}")
            visit(child, depth + 1)

    visit(tree.root, 0)
    if best['p'] is None:
        return None, math.inf
    return best['p'], math.sqrt(best['d2'])

# Program Design - Method 3: Branch-and-Bound with divide Divide and Concquer

def build_quadrant_trees(points, B):
    """Split the data space into 4 rectangles at the midpoint of the X range and of
    the Y range; build one R-tree per non-empty rectangle with the SHARED module."""
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    mid_x = (min(xs) + max(xs)) / 2
    mid_y = (min(ys) + max(ys)) / 2
    quads = [[], [], [], []]
    for p in points:                                # file order is preserved in each quadrant
        i = (0 if p['x'] <= mid_x else 1) + (0 if p['y'] <= mid_y else 2)
        quads[i].append(p)
    trees = []
    for pts in quads:
        if pts:                                     # an empty quadrant has no tree to search
            t = RTree(B=B)
            for p in pts:
                t.insert_point(p)
            trees.append(t)
    return trees, (mid_x, mid_y), [len(pts) for pts in quads]

def bab_divide_and_conquer(trees, q):
    """Run BaB independently on each quadrant tree, keep the closest of the candidates."""
    best_p, best_d2 = None, math.inf
    for t in trees:
        p, _ = bab_nn_search(t, q)
        if p is None:
            continue
        d2 = dist2(q, p)
        if best_p is None or is_better(d2, p, best_d2, best_p):
            best_p, best_d2 = p, d2
    if best_p is None:
        return None, math.inf
    return best_p, math.sqrt(best_d2)

# Running the code + output

def run_method(search_fn, queries):
    """Time ONLY the nearest-neighbour search over all queries."""
    start = time.perf_counter()
    results = [search_fn(q) for q in queries]
    elapsed = time.perf_counter() - start
    return results, elapsed


def format_section(title, queries, results, elapsed):
    lines = [f"=== {title} ==="]
    for q, (p, d) in zip(queries, results):
        lines.append(f"Query {q['id']}: id={p['id']}, x={p['x']}, y={p['y']}, distance={d:.2f}")
    lines.append(f"Total running time for {len(queries)} queries: {elapsed:.6f} seconds")
    return "\n".join(lines)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_B
    points = load_points(PARKING)
    queries = load_points(QUERIES)

    # tree construction (NOT timed)
    tree = RTree(B=B)
    for p in points:
        tree.insert_point(p)
    quad_trees, (mid_x, mid_y), sizes = build_quadrant_trees(points, B)

    seq_res, seq_t = run_method(lambda q: sequential_scan(points, q), queries)
    bab_res, bab_t = run_method(lambda q: bab_nn_search(tree, q), queries)
    dc_res, dc_t = run_method(lambda q: bab_divide_and_conquer(quad_trees, q), queries)

    for name, res in (("BaB", bab_res), ("BaB+D&C", dc_res)):
        bad = [q['id'] for q, a, b in zip(queries, seq_res, res)
               if a[0]['id'] != b[0]['id'] or abs(a[1] - b[1]) > 1e-9]
        print(f"Check {name} vs sequential scan: "
              f"{'all ' + str(len(queries)) + ' queries match' if not bad else 'MISMATCH on queries ' + str(bad)}")
    print(f"B={B}, quadrant midpoints=({mid_x}, {mid_y}), quadrant sizes={sizes}\n")

    summary = "\n".join([
        "=== Running Time Summary ===",
        f"Sequential Scan                     : {seq_t:.6f} seconds",
        f"BaB (B={B})                         : {bab_t:.6f} seconds",
        f"BaB with Divide-and-Conquer (B={B}) : {dc_t:.6f} seconds",
        f"Speed-up of BaB over sequential scan: {seq_t / bab_t:.1f}x",
        f"Speed-up of BaB+D&C over sequential : {seq_t / dc_t:.1f}x",
    ])

    report = "\n\n".join([
        format_section("Sequential Scan", queries, seq_res, seq_t),
        format_section(f"Branch-and-Bound (BaB), B={B}", queries, bab_res, bab_t),
        format_section(f"BaB with Divide-and-Conquer (2x2 subspaces), B={B}", queries, dc_res, dc_t),
        summary,
    ])
    
    print(report)
    with open(OUTPUT, "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
