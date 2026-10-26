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
    return 2 * ((box[2] - box[0]) + (box[3] - box[1]))

