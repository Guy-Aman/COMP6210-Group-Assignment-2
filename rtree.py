import math

def mbr_of_points(points):
    """points: list of (id, x, y). Returns (smin, ymin, xmax, ymax)"""
    x_points = [point[1] for point in points]
    y_points = [point[2] for point in points]
    
    return (min(x_points), min(y_points), max(x_points), max(y_points))

def mbr_of_nodes(nodes):
    """nodes: list of child Nodes. Returns the MBR covering all their MBRS"""
    return (min(n.mbr[0] for n in nodes),
            min(n.mbr[1] for n in nodes),
            max(n.mbr[2] for n in nodes),
            max(n.mbr[3] for n in nodes))

def perimeter(box):
    width = box[2] - box[0] 
    height = box[3] - box[1]
    return 2 * (width + height)

class Node:
    __slots__ = ("leaf", "entries", "mbr", "parent")
    
    def __init__(self, leaf):
        self.leaf = leaf
        self.entries = []
        self.mbr = None
        self.parent = None

# node = Node(True)