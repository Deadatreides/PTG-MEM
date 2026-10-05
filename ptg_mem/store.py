# -*- coding: utf-8 -*-
"""Store format 6: the graph on disk, fast enough for a background daemon.

The engine's own format keeps every vector-like thing as JSON text. On the
reference archive (22 405 atoms, 2560-d) that is 1.7 GB, of which 1.06 GB are
branch centroids written one number per line — about 30 bytes per float instead
of 4 — and a full save took ~90 s. A daemon that saves after every burst of edits
cannot live with that.

Format 6 keeps the same objects but splits index from data (the ZIP central
directory / ISAM principle, measured earlier as 7.4x smaller and 1340x faster to
load for centroids):

    meta.json               id_order, processed_files, embedder, path memory, format
    nodes.json              nodes without vectors (compact JSON)
    vectors.npy             float32 [n, dim] in id_order
    edges.json              typed edges
    branches.json           branch heads/roots
    branch_state.json       momentum / activation / entropy
    branch_bodies.json      bodies without centroids
    branch_centroids.npy    float32 [b, dim] + branch_centroid_ids.json
    files.json              per-file info without centroids
    file_centroids.npy      float32 [f, dim] + file_centroid_ids.json
    file_edges.json         file graph
    root_project.json, root_embedding.npy

A save writes a complete new directory next to the old one and swaps them, so a
crash mid-save leaves the previous generation intact.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

FORMAT = 6
DATA = "data"


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stack(rows, dim):
    if not rows:
        return np.zeros((0, dim or 0), dtype="float32")
    return np.ascontiguousarray(np.asarray(rows, dtype="float32"))


def exists(store_dir: str) -> bool:
    return os.path.isfile(os.path.join(store_dir, DATA, "meta.json"))


def save(arc, store_dir: str, extra: dict | None = None) -> dict:
    """Write the archive ``arc`` (a ptg_core.Archive) as format 6. Returns timings."""
    t0 = time.time()
    os.makedirs(store_dir, exist_ok=True)
    new = os.path.join(store_dir, DATA + ".new")
    cur = os.path.join(store_dir, DATA)
    old = os.path.join(store_dir, DATA + ".old")
    if os.path.exists(new):
        shutil.rmtree(new)
    os.makedirs(new)

    dim = arc.embedder.dim or 0
    ids = list(arc.id_order)
    nodes_out, vecs = {}, []
    for nid in ids:
        n = arc.nodes[nid]
        vecs.append(n["vec"])
        nodes_out[nid] = {k: v for k, v in n.items() if k != "vec"}
    if vecs and not dim:
        dim = int(np.asarray(vecs[0]).shape[0])
    np.save(os.path.join(new, "vectors.npy"), _stack(vecs, dim))
    _dump(os.path.join(new, "nodes.json"), nodes_out)
    _dump(os.path.join(new, "edges.json"), arc.edges)
    _dump(os.path.join(new, "branches.json"), dict(arc.branches))
    _dump(os.path.join(new, "branch_state.json"), arc.branch_states)

    b_ids, b_rows, bodies = [], [], {}
    for bid, body in arc.branch_bodies.items():
        c = body.get("centroid")
        bodies[bid] = {k: v for k, v in body.items() if k != "centroid"}
        if c is not None:
            b_ids.append(bid)
            b_rows.append(c)
    _dump(os.path.join(new, "branch_bodies.json"), bodies)
    np.save(os.path.join(new, "branch_centroids.npy"), _stack(b_rows, dim))
    _dump(os.path.join(new, "branch_centroid_ids.json"), b_ids)

    f_ids, f_rows, files = [], [], {}
    for path, fd in arc.files.items():
        c = fd.get("centroid")
        files[path] = {k: v for k, v in fd.items() if k != "centroid"}
        if c is not None and len(c):
            f_ids.append(path)
            f_rows.append(c)
    _dump(os.path.join(new, "files.json"), files)
    np.save(os.path.join(new, "file_centroids.npy"), _stack(f_rows, dim))
    _dump(os.path.join(new, "file_centroid_ids.json"), f_ids)
    _dump(os.path.join(new, "file_edges.json"), arc.file_edges)
    _dump(os.path.join(new, "root_project.json"), arc.root_project)
    if arc.root_embedding is not None:
        np.save(os.path.join(new, "root_embedding.npy"), np.asarray(arc.root_embedding, dtype="float32"))

    meta = {
        "format": FORMAT,
        "saved_at": time.time(),
        "id_order": ids,
        "processed_files": arc.processed_files,
        "last_structure_hash": arc.last_structure_hash,
        "embedder_backend": arc.embedder.backend,
        "embedder_model": arc.embedder.model,
        "embedder_dim": dim,
        "path_memory": arc.path_memory,
        "atoms_since_decay": arc._atoms_since_decay,
        "prev_branch_id": arc._prev_branch_id,
        "total_nodes": len(ids),
        "total_branches": len(arc.branches),
    }
    if extra:
        meta["extra"] = extra
    _dump(os.path.join(new, "meta.json"), meta)          # written last: marks completeness

    if os.path.exists(old):
        shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(cur):
        os.replace(cur, old)
    os.replace(new, cur)
    shutil.rmtree(old, ignore_errors=True)
    return {"seconds": round(time.time() - t0, 2), "nodes": len(ids)}


def load(arc, store_dir: str) -> dict:
    """Fill ``arc`` from format 6. Returns the ``extra`` dict saved with it."""
    d = os.path.join(store_dir, DATA)
    if not exists(store_dir) and os.path.isdir(os.path.join(store_dir, DATA + ".old")):
        # a crash between the two renames of a save: the previous generation survives
        os.replace(os.path.join(store_dir, DATA + ".old"), d)
    meta = _load(os.path.join(d, "meta.json"))
    if not meta or meta.get("format") != FORMAT:
        raise ValueError("not a format-%d store: %s" % (FORMAT, store_dir))
    ids = meta["id_order"]
    nodes = _load(os.path.join(d, "nodes.json"), {})
    vecs = np.load(os.path.join(d, "vectors.npy"))
    if len(vecs) != len(ids):
        raise ValueError("store is inconsistent: %d ids, %d vectors" % (len(ids), len(vecs)))
    arc.nodes = {}
    for i, nid in enumerate(ids):
        n = nodes[nid]
        n.setdefault("status", "active")
        n["vec"] = vecs[i]
        arc.nodes[nid] = n
    arc.id_order = list(ids)
    arc.edges = _load(os.path.join(d, "edges.json"), [])
    arc.branches = _load(os.path.join(d, "branches.json"), {})
    arc.branch_states = _load(os.path.join(d, "branch_state.json"), {})

    bodies = _load(os.path.join(d, "branch_bodies.json"), {})
    b_ids = _load(os.path.join(d, "branch_centroid_ids.json"), [])
    b_mat = np.load(os.path.join(d, "branch_centroids.npy"))
    for bid, body in bodies.items():
        body["centroid"] = None
    for i, bid in enumerate(b_ids):
        if bid in bodies:
            bodies[bid]["centroid"] = b_mat[i]
    arc.branch_bodies = bodies

    files = _load(os.path.join(d, "files.json"), {})
    f_ids = _load(os.path.join(d, "file_centroid_ids.json"), [])
    f_mat = np.load(os.path.join(d, "file_centroids.npy"))
    for i, path in enumerate(f_ids):
        if path in files:
            files[path]["centroid"] = f_mat[i]
    arc.files = files
    arc.file_edges = _load(os.path.join(d, "file_edges.json"), [])
    arc.root_project = _load(os.path.join(d, "root_project.json"), {})
    rp = os.path.join(d, "root_embedding.npy")
    arc.root_embedding = np.load(rp) if os.path.exists(rp) else None

    arc.processed_files = meta.get("processed_files", {})
    arc.last_structure_hash = meta.get("last_structure_hash")
    arc.embedder_model_used = meta.get("embedder_model")
    arc.embedder_dim_used = meta.get("embedder_dim")
    arc.path_memory = meta.get("path_memory") or {"short": [], "long": []}
    arc._atoms_since_decay = meta.get("atoms_since_decay", 0)
    arc._prev_branch_id = meta.get("prev_branch_id")
    return meta.get("extra") or {}


def load_legacy(arc, legacy_store: str) -> None:
    """Load a store written by ptg_core itself (``<store>/.ptg/*.json``)."""
    import ptg_core
    prev = ptg_core.MMAP_THRESHOLD_BYTES
    ptg_core.MMAP_THRESHOLD_BYTES = 1 << 62          # never memory-map: it blocks later saves
    try:
        arc.ptg_dir = os.path.join(legacy_store, ptg_core.PTG_DIRNAME)
        if not arc.load_if_exists():
            raise ValueError("no legacy store in %s" % legacy_store)
    finally:
        ptg_core.MMAP_THRESHOLD_BYTES = prev
    # files[*].centroid is a JSON list there; keep one representation in memory
    for fd in arc.files.values():
        c = fd.get("centroid")
        if isinstance(c, list):
            fd["centroid"] = np.asarray(c, dtype="float32") if c else None
    for body in arc.branch_bodies.values():
        c = body.get("centroid")
        if isinstance(c, list):
            body["centroid"] = np.asarray(c, dtype="float32")
