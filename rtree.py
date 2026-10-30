"""
Tree structure available to every task:
    tree.root                 root Node
    node.is_leaf()            True for leaf nodes
    node.child_nodes          children (internal nodes)
    node.data_points          points (leaf nodes), each {'id','x','y'}
    node.MBR                  {'x1','y1','x2','y2'}
    node.parent, node.id

Insert points in FILE ORDER so everyone gets the identical tree.

    from rtree_shared import RTree, load_points
    tree = RTree(B=5, verbose=True)
    for p in load_points("Sample_Data/skyline_search.txt"):
        tree.insert_point(p)
"""
import sys
import math


class Node(object):
    def __init__(self, B=4):              # EDIT: B passed in (no global B)
        self.B = B
        self.id = 0
        self.child_nodes = []
        self.data_points = []
        self.parent = None
        # EDIT: empty MBR uses +/-inf instead of -1 (-1 corrupted x1/y1 of new nodes)
        self.MBR = {
            'x1': math.inf,
            'y1': math.inf,
            'x2': -math.inf,
            'y2': -math.inf,
        }

    def perimeter(self):
        return (self.MBR['x2'] - self.MBR['x1']) + (self.MBR['y2'] - self.MBR['y1'])

    def is_overflow(self):
        if self.is_leaf():
            return len(self.data_points) > self.B     # EDIT: uses self.B
        else:
            return len(self.child_nodes) > self.B     # EDIT: uses self.B

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.child_nodes) == 0


class RTree(object):
    def __init__(self, B=4, verbose=False):           # EDIT: configurable B, trace log
        self.B = B
        self.verbose = verbose
        self.log = []          # EDIT: step-by-step trace for the report
        self._next_id = 1      # EDIT: node ids (N1, N2, ...) for diagrams
        self.root = self._new_node()

    # ---------- EDIT: helpers added ----------
    def _new_node(self):
        n = Node(self.B)
        n.id = self._next_id
        self._next_id += 1
        return n

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg)

    @staticmethod
    def _pt(p):
        return f"P{p['id']}({p['x']},{p['y']})"

    @staticmethod
    def _mbr(n):
        m = n.MBR
        return f"[({m['x1']},{m['y1']})-({m['x2']},{m['y2']})]"

    def insert_point(self, point):
        """Insert one point starting from the root (use this from outside)."""
        self._say(f"Insert {self._pt(point)}")
        self.insert(self.root, point)

    # ---------- original teacher code (with edits marked) ----------
    def insert(self, node, point):
        if node.is_leaf():
            self.add_data_point(node, point)
            self._say(f"  -> leaf N{node.id} now {[self._pt(p) for p in node.data_points]}")
            if node.is_overflow():
                self.handle_overflow(node)
        else:
            sub_node = self.choose_subtree(node, point)
            self.insert(sub_node, point)
            self.update_mbr(sub_node)
            self.update_mbr(node)      # EDIT: also refresh this node's MBR (root MBR went stale)

    def add_data_point(self, node, data_point):
        node.data_points.append(data_point)
        if data_point['x'] < node.MBR['x1']:
            node.MBR['x1'] = data_point['x']
        if data_point['x'] > node.MBR['x2']:
            node.MBR['x2'] = data_point['x']
        if data_point['y'] < node.MBR['y1']:
            node.MBR['y1'] = data_point['y']
        if data_point['y'] > node.MBR['y2']:
            node.MBR['y2'] = data_point['y']

    def add_child(self, node, child):
        node.child_nodes.append(child)
        child.parent = node
        if child.MBR['x1'] < node.MBR['x1']:
            node.MBR['x1'] = child.MBR['x1']
        if child.MBR['x2'] > node.MBR['x2']:
            node.MBR['x2'] = child.MBR['x2']
        if child.MBR['y1'] < node.MBR['y1']:
            node.MBR['y1'] = child.MBR['y1']
        if child.MBR['y2'] > node.MBR['y2']:
            node.MBR['y2'] = child.MBR['y2']

    def handle_overflow(self, node):
        self._say(f"  OVERFLOW at N{node.id} (more than B={self.B} entries)")
        node1, node2 = self.split(node)
        if node.is_root():
            new_root = self._new_node()
            self.add_child(new_root, node1)
            self.add_child(new_root, node2)
            self.root = new_root
            self.update_mbr(new_root)
            self._say(f"  ROOT SPLIT -> new root N{new_root.id} {self._mbr(new_root)}")
        else:
            parent = node.parent
            parent.child_nodes.remove(node)
            self.add_child(parent, node1)
            self.add_child(parent, node2)
            self.update_mbr(parent)    # EDIT: recompute exactly (add_child only expands)
            if parent.is_overflow():
                self.handle_overflow(parent)

    def choose_subtree(self, node, point):
        if node.is_leaf():
            return node
        else:
            min_increase = sys.maxsize
            best_child = None
            for child in node.child_nodes:
                inc = self.peri_increase(child, point)
                if min_increase > inc:
                    min_increase = inc
                    best_child = child
            return best_child

    def peri_increase(self, node, point):
        origin_mbr = node.MBR
        x1, x2, y1, y2 = origin_mbr['x1'], origin_mbr['x2'], origin_mbr['y1'], origin_mbr['y2']
        increase = (max([x1, x2, point['x']]) - min([x1, x2, point['x']]) +
                    max([y1, y2, point['y']]) - min([y1, y2, point['y']])) - node.perimeter()
        return increase

    def split(self, node):
        best_s1 = self._new_node()     # EDIT: nodes created with B and an id
        best_s2 = self._new_node()
        best_perimeter = sys.maxsize
        lo = max(1, math.ceil(0.4 * self.B))          # EDIT: uses self.B
        if node.is_leaf():
            m = len(node.data_points)
            divides = [sorted(node.data_points, key=lambda d: d['x']),
                       sorted(node.data_points, key=lambda d: d['y'])]
            for divide in divides:
                for i in range(lo, m - lo + 1):
                    s1 = Node(self.B)
                    s1.data_points = divide[0:i]
                    self.update_mbr(s1)
                    s2 = Node(self.B)
                    s2.data_points = divide[i:len(divide)]
                    self.update_mbr(s2)
                    if best_perimeter > s1.perimeter() + s2.perimeter():
                        best_perimeter = s1.perimeter() + s2.perimeter()
                        best_s1.data_points, best_s1.MBR = s1.data_points, s1.MBR   # EDIT: keep id of reserved nodes
                        best_s2.data_points, best_s2.MBR = s2.data_points, s2.MBR
        else:
            m = len(node.child_nodes)
            divides = [sorted(node.child_nodes, key=lambda c: c.MBR['x1']),
                       sorted(node.child_nodes, key=lambda c: c.MBR['x2']),
                       sorted(node.child_nodes, key=lambda c: c.MBR['y1']),
                       sorted(node.child_nodes, key=lambda c: c.MBR['y2'])]
            for divide in divides:
                for i in range(lo, m - lo + 1):
                    s1 = Node(self.B)
                    s1.child_nodes = divide[0:i]
                    self.update_mbr(s1)
                    s2 = Node(self.B)
                    s2.child_nodes = divide[i:len(divide)]
                    self.update_mbr(s2)
                    if best_perimeter > s1.perimeter() + s2.perimeter():
                        best_perimeter = s1.perimeter() + s2.perimeter()
                        best_s1.child_nodes, best_s1.MBR = s1.child_nodes, s1.MBR
                        best_s2.child_nodes, best_s2.MBR = s2.child_nodes, s2.MBR

        for child in best_s1.child_nodes:
            child.parent = best_s1
        for child in best_s2.child_nodes:
            child.parent = best_s2

        def desc(n):
            return [self._pt(p) for p in n.data_points] if n.is_leaf() else [f"N{c.id}" for c in n.child_nodes]
        self._say(f"  SPLIT N{node.id} -> N{best_s1.id}{desc(best_s1)} {self._mbr(best_s1)} | "
                  f"N{best_s2.id}{desc(best_s2)} {self._mbr(best_s2)}")
        return best_s1, best_s2

    def update_mbr(self, node):
        if node.is_leaf():
            x_list = [p['x'] for p in node.data_points]
            y_list = [p['y'] for p in node.data_points]
        else:
            x_list = [c.MBR['x1'] for c in node.child_nodes] + [c.MBR['x2'] for c in node.child_nodes]
            y_list = [c.MBR['y1'] for c in node.child_nodes] + [c.MBR['y2'] for c in node.child_nodes]
        if not x_list:                 # EDIT: empty node guard
            return
        node.MBR = {'x1': min(x_list), 'x2': max(x_list),
                    'y1': min(y_list), 'y2': max(y_list)}

    # ---------- EDIT: inspection helper for report diagrams ----------
    def dump(self, node=None, indent=0):
        node = node or self.root
        kind = "leaf" if node.is_leaf() else "internal"
        print("  " * indent + f"N{node.id} ({kind}) {self._mbr(node)}")
        if node.is_leaf():
            for p in node.data_points:
                print("  " * (indent + 1) + self._pt(p))
        else:
            for c in node.child_nodes:
                self.dump(c, indent + 1)


# ---------- EDIT: shared data loader (floats, optional id, commas) ----------
def load_points(path):
    """Accepts lines 'id x y' or 'x y' (spaces or commas). Returns list of dicts."""
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) == 3:
                pid, x, y = int(float(parts[0])), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                pid, x, y = len(pts) + 1, float(parts[0]), float(parts[1])
            else:
                continue
            pts.append({'id': pid, 'x': x, 'y': y})
    return pts


if __name__ == "__main__":
    # quick self-test: build a tree with B=5 and print the construction steps
    pts = [(2,9),(4,8),(3,6),(6,7),(1,10),(7,3),(5,5),(9,2),(8,6),(10,1),(6,4),(3,8),(9,9),(4,3),(7,7)]
    tree = RTree(B=5, verbose=True)
    for i, (x, y) in enumerate(pts, 1):
        tree.insert_point({'id': i, 'x': x, 'y': y})
    print("\nFinal tree:")
    tree.dump()