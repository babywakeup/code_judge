"""Cyclic identity layout selection and directed spacing learned from references.

Candidates contain private array fields (such as ``_mask``); these dictionaries
are internal evidence and must be stripped before exporting diagnostics.
"""

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np

from ...utils.types import Image, Matrix


Candidate: TypeAlias = dict[str, Any]
CandidatePools: TypeAlias = dict[str, dict[str, list[Candidate]]]
Spacing: TypeAlias = dict[tuple[str, str], float]
Layout: TypeAlias = tuple[float, list[str], list[Candidate]]


def _quality(candidate: Candidate) -> float:
    """Prefer whole vertical structures over a coincidental short end fragment.

    Horizontal clipping at the input edge remains allowed and lightly penalized;
    it must not make an unrelated full-width identity win solely by area.
    """
    return (candidate['score']-.2*(1-candidate['vertical_fraction'])
            -.025*(1-candidate['horizontal_fraction'])-.05*candidate['reference_partial'])


def solve_layout(pools: CandidatePools, cycle: list[str], spacing: Spacing,
                 spacing_weight: float, visible_count: int | None = None) -> list[Layout]:
    """Best path for each cyclic rotation, with learned directed neighbor gaps.

    Different source references for the same occurrence do not create a second
    phase. Keeping two paths per endpoint also supports N=1 location ambiguity.
    Spacing is a soft constraint; no equal-spacing or image-width assumption.
    """
    n = len(cycle) if visible_count is None else visible_count
    bounds = {}
    for pool in pools.values():
        for c in pool['distinct']:
            ys, xs = np.where(c['_mask'])
            bounds[id(c)] = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1, len(xs))
    cache = {}
    def transition(a: Candidate, b: Candidate) -> float | None:
        key = (id(a), id(b))
        if key not in cache:
            if a['center_x'] >= b['center_x']:
                cache[key] = None
            else:
                ax0, ay0, ax1, ay1, aa = bounds[id(a)]
                bx0, by0, bx1, by1, ba = bounds[id(b)]
                x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
                overlap = ((a['_mask'][y0:y1, x0:x1] & b['_mask'][y0:y1, x0:x1]).sum()
                           if x1 > x0 and y1 > y0 else 0)
                expected = spacing.get((a['identity'], b['identity']))
                cost = (spacing_weight*min(abs((b['center_x']-a['center_x'])/expected-1), 1)/max(n-1, 1)
                        if expected else 0.)
                cache[key] = cost if overlap <= .15*min(aa, ba) else None
        return cache[key]
    groups = []
    for phase in range(len(cycle)):
        order = (cycle[phase:]+cycle[:phase])[:n]
        paths = [(_quality(c)/n, [c]) for c in pools[order[0]]['distinct']]
        for name in order[1:]:
            next_paths = []
            for c in pools[name]['distinct']:
                options = []
                for score, path in paths:
                    cost = transition(path[-1], c)
                    if cost is not None:
                        options.append((score+_quality(c)/n-cost, path+[c]))
                next_paths.extend(sorted(options, key=lambda p: p[0], reverse=True)[:2])
            paths = next_paths
        for score, path in sorted(paths, key=lambda p: p[0], reverse=True)[:2 if n == 1 else 1]:
            groups.append((score, order, path))
    return sorted(groups, key=lambda g: g[0], reverse=True)


def _template_spacings(
    images: Sequence[Image], references: Sequence[Sequence[Candidate]],
    matrices: Sequence[Matrix | Sequence[Sequence[float]] | None], cycle: Sequence[str],
) -> Spacing:
    """Directed neighboring gaps measured from original masks, in target pixels."""
    observations = {}
    for image, refs, H in zip(images, references, matrices):
        if H is None:
            continue
        centers = []
        for ref in refs:
            yy, xx = np.where(ref['mask'] > 0)
            if not len(xx):
                continue
            # A source-edge fragment has a biased center; use a complete reference
            # from another template rather than learning that bias as a spacing.
            if xx.min() < 2 or xx.max() >= image.shape[1]-2:
                continue
            p = np.array([(xx.min()+xx.max())/2, (yy.min()+yy.max())/2, 1.])
            centers.append((p[0], ref['identity'], float((np.asarray(H) @ p)[0])))
        centers.sort()
        for (_, a, x), (_, b, y) in zip(centers, centers[1:]):
            if b == cycle[(cycle.index(a)+1) % len(cycle)] and y > x:
                observations.setdefault((a, b), []).append(y-x)
    return {key: float(np.median(values)) for key, values in observations.items()}

