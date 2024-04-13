import os
import subprocess
import numpy as np

from tempfile import TemporaryDirectory


def _dot_var(v, verbose=False):
    # Format string for a Variable node
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name

    # Label format in verbose=True: "<name>: <shape> <data type>"
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)

    return dot_var.format(id(v), name)  # id() returns a unique object ID


def _dot_func(f):
    # Format string for a Function node
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    # Edges between Function and its I/O Variables
    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id((y())))  # y is weakref
    return txt


def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    # Add Function and Variable nodes recursively
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    # Save DOT data to a temporary directory
    # tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    # if not os.path.exists(tmp_dir):
    #     os.mkdir(tmp_dir)
    # graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with TemporaryDirectory() as tmp_dir:  # Use tempfile module
        graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

        with open(graph_path, "w") as f:
            f.write(dot_graph)

        # Call dot command
        extension = os.path.splitext(to_file)[1][1:]
        cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
        subprocess.run(cmd, shell=True)


# Sum elements along axes to output an array of a given shape
def sum_to(x, shape):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


# Reshape gradient appropriately for dezero.funcions.sum's backward
def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    # Restore a shape used for backward
    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)
    return gy


def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += 2
    return m
