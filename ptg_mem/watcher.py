# -*- coding: utf-8 -*-
"""Change detection with a debounce queue.

``watchdog`` (inotify / FSEvents / ReadDirectoryChangesW) when it is installed;
otherwise a polling scan whose interval adapts to how long a scan takes, so a
huge tree is not rescanned back-to-back. Either way a periodic full rescan runs
as a safety net: file-system events are lost on network drives, during sleep,
and when an editor saves by rename.

Events are debounced per path: an editor that saves five times in a second, or
an agent appending to its transcript on every token, produces one ingest after
the source has been quiet for ``delay`` seconds.
"""
from __future__ import annotations

import heapq
import os
import threading
import time

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    HAVE_WATCHDOG = True
except ImportError:                                   # pragma: no cover
    HAVE_WATCHDOG = False
    FileSystemEventHandler = object


class DebounceQueue:
    def __init__(self):
        self._due: dict[tuple, float] = {}
        self._heap: list = []
        self._cv = threading.Condition()

    def put(self, key: tuple, delay: float):
        with self._cv:
            due = time.time() + delay
            self._due[key] = due
            heapq.heappush(self._heap, (due, key))
            self._cv.notify()

    def get(self, timeout: float = 1.0):
        """Next key whose quiet period has passed, or None after ``timeout``."""
        end = time.time() + timeout
        with self._cv:
            while True:
                now = time.time()
                while self._heap:
                    due, key = self._heap[0]
                    if self._due.get(key) != due:          # superseded by a later put
                        heapq.heappop(self._heap)
                        continue
                    if due <= now:
                        heapq.heappop(self._heap)
                        del self._due[key]
                        return key
                    break
                wait = end - now
                if self._heap:
                    wait = min(wait, self._heap[0][0] - now)
                if wait <= 0:
                    return None
                self._cv.wait(wait)

    def __len__(self):
        with self._cv:
            return len(self._due)

    def pending(self) -> list:
        with self._cv:
            return sorted(self._due, key=lambda k: self._due[k])


class _Handler(FileSystemEventHandler):
    def __init__(self, on_path):
        self.on_path = on_path

    def on_any_event(self, event):                    # noqa: D401
        if getattr(event, "is_directory", False):
            return
        for attr in ("src_path", "dest_path"):
            p = getattr(event, attr, None)
            if p:
                self.on_path(os.fsdecode(p))


class Watcher:
    """Watches a set of folders and calls ``on_path(path)`` for every change."""

    def __init__(self, folders: list[str], on_path, poll_fn=None, log=None):
        self.folders = [f for f in folders if f and os.path.isdir(f)]
        self.on_path = on_path
        self.poll_fn = poll_fn                        # full rescan: returns nothing, enqueues itself
        self.log = log or (lambda m: None)
        self._obs = None
        self._stop = threading.Event()
        self._thread = None
        self.mode = "watchdog" if HAVE_WATCHDOG else "polling"

    def start(self):
        if HAVE_WATCHDOG:
            try:
                self._obs = Observer()
                h = _Handler(self.on_path)
                for f in self.folders:
                    self._obs.schedule(h, f, recursive=True)
                self._obs.daemon = True
                self._obs.start()
            except Exception as ex:                   # noqa: BLE001
                self.log("watchdog failed (%s), falling back to polling" % ex)
                self._obs = None
                self.mode = "polling"
        self._thread = threading.Thread(target=self._rescan_loop, daemon=True, name="ptg-rescan")
        self._thread.start()
        return self

    def _rescan_loop(self):
        interval = 600.0 if self._obs else 20.0
        while not self._stop.wait(interval):
            if not self.poll_fn:
                continue
            t0 = time.time()
            try:
                self.poll_fn()
            except Exception as ex:                   # noqa: BLE001
                self.log("rescan failed: %s" % ex)
            took = time.time() - t0
            if not self._obs:
                # a scan must not eat more than ~5 % of the machine
                interval = max(20.0, 20.0 * took)

    def stop(self):
        self._stop.set()
        if self._obs:
            try:
                self._obs.stop()
                self._obs.join(timeout=3)
            except Exception:                         # noqa: BLE001
                pass
