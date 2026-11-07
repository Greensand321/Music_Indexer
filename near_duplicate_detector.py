# near_duplicate_detector.py
"""Helpers for fuzzy fingerprint grouping."""
from __future__ import annotations

import os
import re
import hashlib
import struct
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Set, List, Iterable, Tuple, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import base64
from itertools import combinations

logger = logging.getLogger(__name__)

EXCLUSION_KEYWORDS = ['remix', 'remastered', 'edit', 'version']
COARSE_FP_BAND_SIZE = 8
COARSE_FP_BANDS = 6
COARSE_FP_QUANTIZATION = 4
LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".ape", ".aiff", ".aif"}


def _keep_score(path: str, info: Mapping[str, object], ext_priority: Mapping[str, int]) -> float:
    """Compute keep score for a file based on extension, metadata and filename."""
    ext = os.path.splitext(path)[1].lower()
    pri = ext_priority.get(ext, 99)
    ext_score = 1000.0 / (pri + 1)
    meta_score = int(info.get("meta_count", 0) or 0) * 10
    fname_score = len(os.path.splitext(os.path.basename(path))[0])
    return ext_score + meta_score + fname_score


@dataclass
class NearDuplicateGroup:
    winner: str
    losers: List[str]
    max_distance: float
    reason: str


@dataclass
class NearDuplicateResult:
    auto_deletes: Dict[str, str]
    review_groups: List[NearDuplicateGroup]

    @property
    def review_required(self) -> bool:
        return bool(self.review_groups)

    def items(self):
        return self.auto_deletes.items()


def _fp_text(fp: object) -> str | None:
    """Return ``fp`` as text. pyacoustid / libchromaprint hand back ``bytes``."""
    if fp is None:
        return None
    if isinstance(fp, (bytes, bytearray, memoryview)):
        try:
            return bytes(fp).decode("ascii")
        except UnicodeDecodeError:
            return bytes(fp).decode("latin1")
    if isinstance(fp, str):
        return fp
    return str(fp)


# ── Chromaprint fingerprint decoding ─────────────────────────────────────────
#
# ``fpcalc -json`` (no ``-raw``), ``acoustid.fingerprint_file`` and the
# fingerprint caches all carry the *compressed* Chromaprint fingerprint: a
# URL-safe base64 string of a delta/bit-position encoded stream. Comparing
# those bytes directly is meaningless — one flipped bit in the underlying
# fingerprint shifts every byte after it, so a FLAC and its own MP3 scored
# ~0.23 instead of ~0.0. ``decode_chromaprint`` undoes the compression
# (a pure-Python port of chromaprint's FingerprintDecompressor, so it works
# whether or not libchromaprint is installed) and ``fingerprint_distance``
# then compares the raw 32-bit sub-fingerprints bit by bit.

_CP_NORMAL_BITS = 3
_CP_EXCEPTION_BITS = 5
_CP_MAX_NORMAL = (1 << _CP_NORMAL_BITS) - 1


def _read_packed(data: bytes, start: int, width: int, count: int) -> List[int] | None:
    """Read ``count`` little-endian ``width``-bit values from ``data[start:]``."""
    out: List[int] = []
    acc = nbits = 0
    pos = start
    mask = (1 << width) - 1
    end = len(data)
    while len(out) < count:
        while nbits < width:
            if pos >= end:
                return None
            acc |= data[pos] << nbits
            pos += 1
            nbits += 8
        out.append(acc & mask)
        acc >>= width
        nbits -= width
    return out


def _decompress_chromaprint(data: bytes) -> tuple[int, List[int]] | None:
    """Return ``(algorithm, raw_items)`` for a compressed fingerprint, else None.

    Strict: the payload must be *exactly* what chromaprint's compressor emits
    (header + 3-bit normal stream + 5-bit exception stream), so arbitrary
    base64 text is not mistaken for a fingerprint.
    """
    if len(data) < 5:
        return None
    algorithm = data[0]
    num_values = (data[1] << 16) | (data[2] << 8) | data[3]
    if num_values == 0:
        return None
    normal: List[int] = []
    acc = nbits = 0
    pos = 4
    end = len(data)
    found = exceptions = 0
    while found < num_values:
        if nbits < _CP_NORMAL_BITS:
            if pos >= end:
                return None
            acc |= data[pos] << nbits
            pos += 1
            nbits += 8
        value = acc & _CP_MAX_NORMAL
        acc >>= _CP_NORMAL_BITS
        nbits -= _CP_NORMAL_BITS
        normal.append(value)
        if value == 0:
            found += 1
        elif value == _CP_MAX_NORMAL:
            exceptions += 1
    offset = 4 + (len(normal) * _CP_NORMAL_BITS + 7) // 8
    if end - offset != (exceptions * _CP_EXCEPTION_BITS + 7) // 8:
        return None
    if exceptions:
        extra = _read_packed(data, offset, _CP_EXCEPTION_BITS, exceptions)
        if extra is None:
            return None
        it = iter(extra)
        normal = [v + next(it) if v == _CP_MAX_NORMAL else v for v in normal]
    items: List[int] = []
    value = last_bit = 0
    for bit in normal:
        if bit == 0:
            items.append(value ^ items[-1] if items else value)
            value = last_bit = 0
            continue
        bit += last_bit
        if bit > 32:
            return None
        last_bit = bit
        value |= 1 << (bit - 1)
    return algorithm, items


def _b64decode_strict(text: str) -> bytes | None:
    if not text or not re.fullmatch(r"[A-Za-z0-9_\-+/]+=*", text):
        return None
    try:
        return base64.urlsafe_b64decode(text.replace("+", "-").replace("/", "_") + "=" * (-len(text) % 4))
    except Exception:
        return None


@lru_cache(maxsize=8192)
def _decode_cached(text: str) -> tuple[int, tuple[int, ...]] | None:
    data = _b64decode_strict(text)
    if not data:
        return None
    decoded = _decompress_chromaprint(data)
    if decoded is None:
        return None
    algorithm, items = decoded
    return algorithm, tuple(items)


def is_empty_chromaprint(fp: object) -> bool:
    """True for a Chromaprint fingerprint with zero items (e.g. ``"AQAAAA"``).

    fpcalc emits one for clips too short to fingerprint (about 3 s or less).
    It carries no information, so two of them must not count as identical.
    """
    text = _fp_text(fp)
    if not text:
        return False
    data = _b64decode_strict(text.strip())
    return bool(data) and len(data) == 4 and data[1:4] == b"\x00\x00\x00"


def decode_chromaprint(fp: object) -> tuple[int, List[int]] | None:
    """Decode a compressed Chromaprint fingerprint (str or bytes).

    Returns ``(algorithm, raw_items)`` with unsigned 32-bit items, or ``None``
    when ``fp`` is not a compressed Chromaprint fingerprint.
    """
    text = _fp_text(fp)
    if not text:
        return None
    decoded = _decode_cached(text.strip())
    if decoded is None:
        return None
    return decoded[0], list(decoded[1])


# Raw fingerprints are compared at small relative shifts so that a slightly
# different start (an extra fraction of a second that silence trimming left
# in, a ripper's pregap) does not wreck the comparison. One item is ~0.124 s.
RAW_MAX_ALIGN_OFFSET = 8
RAW_MIN_ALIGN_ITEMS = 32


@lru_cache(maxsize=8192)
def _packed_items(items: tuple[int, ...]) -> bytes:
    return struct.pack(f"<{len(items)}I", *items)


def raw_fingerprint_distance(a: Sequence[int], b: Sequence[int]) -> float:
    """Bit error rate between two raw Chromaprint fingerprints (0.0 – ~0.5).

    Identical audio scores 0.0; transcodes of the same recording score a few
    hundredths; unrelated recordings sit around 0.5 (random bits).
    """
    ta = tuple(int(x) & 0xFFFFFFFF for x in a)
    tb = tuple(int(x) & 0xFFFFFFFF for x in b)
    if not ta or not tb:
        return 1.0
    pa = _packed_items(ta)
    pb = _packed_items(tb)
    shortest = min(len(ta), len(tb))
    max_offset = RAW_MAX_ALIGN_OFFSET if shortest >= RAW_MIN_ALIGN_ITEMS else 0
    min_overlap = max(1, shortest - max_offset) if max_offset else shortest
    best = 1.0
    for offset in range(-max_offset, max_offset + 1):
        sa = offset if offset > 0 else 0
        sb = -offset if offset < 0 else 0
        n = min(len(ta) - sa, len(tb) - sb)
        if n < min_overlap or n <= 0:
            continue
        xa = int.from_bytes(pa[sa * 4:(sa + n) * 4], "little")
        xb = int.from_bytes(pb[sb * 4:(sb + n) * 4], "little")
        ber = (xa ^ xb).bit_count() / (32 * n)
        if ber < best:
            best = ber
            if best == 0.0:
                break
    return best


def _parse_fp(fp: str) -> tuple[str, List[int] | bytes] | None:
    """Parse fingerprint string as integer list or base64 bytes."""
    fp = _fp_text(fp)  # type: ignore[assignment]
    if not fp:
        return None
    text = fp.strip()
    if not text:
        return None
    try:
        arr = [int(x) for x in text.replace(',', ' ').split()]
        if arr:
            return ("ints", arr)
    except Exception:
        pass
    try:
        data = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
        if data:
            return ("bytes", data)
    except Exception:
        pass
    return None


def _plain_ints(text: str) -> List[int] | None:
    try:
        values = [int(x) for x in text.replace(",", " ").split()]
    except ValueError:
        return None
    return values or None


def _hamming_ints(a: List[int], b: List[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = sum(x != y for x, y in zip(a[:n], b[:n]))
    return diff / n


def _hamming_bytes(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = 0
    for i in range(n):
        diff += (a[i] ^ b[i]).bit_count()
    return diff / (n * 8)


def fingerprint_distance(fp1: str | bytes | None, fp2: str | bytes | None) -> float:
    """Return a normalized distance between two fingerprints (0.0 = identical).

    * Compressed Chromaprint fingerprints (what fpcalc ``-json``, pyacoustid
      and the caches produce, as ``str`` or ``bytes``) are decoded and compared
      as raw sub-fingerprints: the bit error rate, best of a few small
      alignments (see ``raw_fingerprint_distance``).
    * A compressed fingerprint against a plain integer list compares the
      integers as raw sub-fingerprints the same way.
    * Two plain integer lists ("1 2 3") keep the historical semantics — the
      fraction of positions that differ — which is what the synthetic test
      fixtures across the suite are written against.
    * Anything else that base64-decodes falls back to the historical bitwise
      Hamming distance over the decoded bytes.
    """
    fp1 = _fp_text(fp1)
    fp2 = _fp_text(fp2)
    if not fp1 or not fp2:
        return 1.0
    fp1 = fp1.strip()
    fp2 = fp2.strip()
    if not fp1 or not fp2:
        return 1.0
    if is_empty_chromaprint(fp1) or is_empty_chromaprint(fp2):
        return 1.0  # "too short to fingerprint" is not evidence of a match
    if fp1 == fp2:
        return 0.0
    raw1 = decode_chromaprint(fp1)
    raw2 = decode_chromaprint(fp2)
    if raw1 is not None or raw2 is not None:
        items1 = raw1[1] if raw1 is not None else _plain_ints(fp1)
        items2 = raw2[1] if raw2 is not None else _plain_ints(fp2)
        if items1 is not None and items2 is not None:
            if raw1 is not None and raw2 is not None and raw1[0] != raw2[0]:
                return 1.0  # different Chromaprint algorithms are not comparable
            return raw_fingerprint_distance(items1, items2)
    p1 = _parse_fp(fp1)
    p2 = _parse_fp(fp2)
    if not p1 or not p2 or p1[0] != p2[0]:
        return 1.0
    kind, a = p1
    _, b = p2
    if kind == "ints":
        return _hamming_ints(a, b)  # type: ignore[arg-type]
    else:
        return _hamming_bytes(a, b)  # type: ignore[arg-type]


def _has_exclusion(info: Dict[str, str | None]) -> bool:
    text = f"{info.get('title','')} {info.get('album','')}".lower()
    return any(k in text for k in EXCLUSION_KEYWORDS)


def _normalized(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _metadata_key(info: Mapping[str, str | None]) -> tuple[str, str, str]:
    return (
        _normalized(info.get("title")),
        _normalized(info.get("primary") or info.get("artist")),
        _normalized(info.get("album")),
    )


def _metadata_gate(info_a: Mapping[str, str | None], info_b: Mapping[str, str | None]) -> bool:
    title_a, primary_a, album_a = _metadata_key(info_a)
    title_b, primary_b, album_b = _metadata_key(info_b)
    if not title_a or not title_b or title_a != title_b:
        return False
    if primary_a and primary_b and primary_a == primary_b:
        return True
    if album_a and album_b and album_a == album_b:
        return True
    return False


def _path_ext(path: str, info: Mapping[str, str | None]) -> str:
    ext = info.get("ext") if isinstance(info, Mapping) else None
    if ext:
        return str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
    return os.path.splitext(path)[1].lower()


def _is_lossless(ext: str) -> bool:
    return ext in LOSSLESS_EXTS


def _pair_threshold(
    path_a: str,
    path_b: str,
    info_a: Mapping[str, str | None],
    info_b: Mapping[str, str | None],
    base_threshold: float,
    mixed_codec_boost: float,
) -> float:
    ext_a = _path_ext(path_a, info_a)
    ext_b = _path_ext(path_b, info_b)
    if not ext_a or not ext_b:
        return base_threshold
    if _is_lossless(ext_a) != _is_lossless(ext_b):
        return base_threshold + mixed_codec_boost
    return base_threshold


COARSE_RAW_ITEMS_PER_BUCKET = 64  # ~8 s of audio per length bucket


def _coarse_fingerprint_keys(fp: str | bytes | None) -> List[str]:
    fp = _fp_text(fp)
    decoded = decode_chromaprint(fp) if fp else None
    if decoded is not None:
        # Real Chromaprint fingerprints: banding the *compressed* bytes split a
        # FLAC from its own low-bitrate MP3 (one flipped bit reshuffles every
        # later byte), so the gate blocked exactly the pairs it should pass.
        # Gate on fingerprint length instead: two keys per print, so prints
        # whose lengths fall in the same or adjacent ~8 s buckets always share
        # one. The real comparison is ``fingerprint_distance``.
        algorithm, items = decoded
        bucket = len(items) // COARSE_RAW_ITEMS_PER_BUCKET
        return [f"cp{algorithm}:len{bucket}", f"cp{algorithm}:len{bucket + 1}"]
    parsed = _parse_fp(fp) if fp else None
    values: List[int] = []
    if parsed:
        kind, payload = parsed
        if kind == "ints":
            values = list(payload)
        elif kind == "bytes":
            values = list(payload)
    if not values and fp:
        values = [ord(ch) for ch in fp]
    if not values:
        return []
    quantized = [val // COARSE_FP_QUANTIZATION for val in values]
    keys: List[str] = []
    limit = COARSE_FP_BAND_SIZE * COARSE_FP_BANDS
    for idx in range(0, min(len(quantized), limit), COARSE_FP_BAND_SIZE):
        band = quantized[idx : idx + COARSE_FP_BAND_SIZE]
        if not band:
            break
        digest = hashlib.blake2s(",".join(map(str, band)).encode("utf-8"), digest_size=4).hexdigest()
        keys.append(f"{idx // COARSE_FP_BAND_SIZE}:{digest}")
    if not keys and fp:
        keys.append(f"band0:{hashlib.blake2s(fp.encode('utf-8'), digest_size=4).hexdigest()}")
    return keys


def _coarse_gate(keys_a: List[str], keys_b: List[str]) -> bool:
    if not keys_a or not keys_b:
        return False
    return bool(set(keys_a) & set(keys_b))


def _cluster_max_distance(cluster: Set[str], distances: Dict[frozenset[str], float]) -> float:
    pairs = [
        distances[frozenset({a, b})]
        for a, b in combinations(sorted(cluster), 2)
        if frozenset({a, b}) in distances
    ]
    return max(pairs) if pairs else 0.0


def _scan_paths(
    paths: Iterable[str],
    file_infos: Dict[str, Dict[str, str | None]],
    threshold: float,
    mixed_codec_boost: float,
    log_callback,
) -> Tuple[List[Tuple[Set[str], float]], int, int, int]:
    """Scan a list of paths for near duplicate clusters."""
    def _pairs_from_index(index: Dict[tuple[str, ...], List[str]]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for members in index.values():
            if len(members) < 2:
                continue
            members_sorted = sorted(members)
            for i, left in enumerate(members_sorted):
                for right in members_sorted[i + 1 :]:
                    pairs.add((left, right))
        return pairs

    adj: Dict[str, Set[str]] = defaultdict(set)
    pair_distances: Dict[frozenset[str], float] = {}
    path_list = list(paths)
    total_pairs = len(path_list) * (len(path_list) - 1) // 2

    metadata_index: Dict[tuple[str, ...], List[str]] = defaultdict(list)
    album_index: Dict[tuple[str, ...], List[str]] = defaultdict(list)
    for p in path_list:
        info = file_infos[p]
        title_key = _normalized(info.get("title"))
        primary_key = _normalized(info.get("primary") or info.get("artist"))
        album_key = _normalized(info.get("album"))
        if title_key and primary_key:
            metadata_index[(title_key, primary_key)].append(p)
        if title_key and album_key:
            album_index[(title_key, album_key)].append(p)

    candidate_pairs = _pairs_from_index(metadata_index) | _pairs_from_index(album_index)
    metadata_blocked = max(0, total_pairs - len(candidate_pairs))
    coarse_keys = {p: _coarse_fingerprint_keys(file_infos[p].get("fp")) for p in path_list}
    comparisons = 0
    coarse_blocked = 0

    for p, q in candidate_pairs:
        if not _coarse_gate(coarse_keys.get(p, []), coarse_keys.get(q, [])):
            coarse_blocked += 1
            continue
        dist = fingerprint_distance(file_infos[p]["fp"], file_infos[q]["fp"])
        comparisons += 1
        pair_threshold = _pair_threshold(
            p,
            q,
            file_infos[p],
            file_infos[q],
            threshold,
            mixed_codec_boost,
        )
        if dist <= pair_threshold:
            adj[p].add(q)
            adj[q].add(p)
            pair_distances[frozenset({p, q})] = dist
            log_callback(
                f"   → near-dup {os.path.basename(p)} vs {os.path.basename(q)} "
                f"dist={dist:.3f} thr={pair_threshold:.3f}"
            )

    visited: Set[str] = set()
    clusters: List[Set[str]] = []
    for p in path_list:
        if p in visited:
            continue
        stack = [p]
        comp = {p}
        visited.add(p)
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    comp.add(nb)
                    stack.append(nb)
        if len(comp) > 1:
            max_distance = _cluster_max_distance(comp, pair_distances)
            clusters.append((comp, max_distance))
    return clusters, comparisons, metadata_blocked, coarse_blocked




def _scan_album(
    album: str,
    paths: Iterable[str],
    file_infos: Dict[str, Dict[str, str | None]],
    threshold: float,
    mixed_codec_boost: float,
    log_callback,
) -> tuple[str, List[Tuple[Set[str], float]], int, int, int]:
    """Worker helper: scan a single album for near duplicates."""
    clusters, comparisons, meta_blocked, coarse_blocked = _scan_paths(
        paths,
        file_infos,
        threshold,
        mixed_codec_boost,
        log_callback,
    )
    return album, clusters, comparisons, meta_blocked, coarse_blocked


def find_near_duplicates(
    file_infos: Dict[str, Dict[str, str | None]],
    ext_priority: Dict[str, int],
    threshold: float,
    log_callback=None,
    enable_cross_album: bool = False,
    coord=None,
    max_workers: int | None = None,
    mixed_codec_boost: float = 0.0,
) -> NearDuplicateResult:
    """Return reviewable near-duplicate groups (auto-merging is gated by review)."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    paths = [p for p, info in file_infos.items() if info.get("fp") and not _has_exclusion(info)]

    by_album: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        album = file_infos[p].get("album") or ""
        by_album[album].append(p)

    album_items = sorted(by_album.items(), key=lambda x: x[0])
    total_albums = len(album_items)
    total_comparisons = 0
    metadata_blocked = 0
    coarse_blocked = 0
    clusters: List[Tuple[Set[str], float]] = []
    html_lines = ["<h2>Phase B – Album Near-Duplicates</h2>", "<pre>"]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for idx, (album, album_paths) in enumerate(album_items, start=1):
            if len(album_paths) < 2:
                continue
            logger.debug(
                f"Phase B: [{idx}/{total_albums}] Scanning album '{album}' ({len(album_paths)} tracks)"
            )
            html_lines.append(f"Album: {album or '<no album>'} ({len(album_paths)} tracks)")
            futures[ex.submit(_scan_album, album, album_paths, file_infos, threshold, mixed_codec_boost, log_callback)] = album

        for fut in as_completed(futures):
            album = futures[fut]
            try:
                _alb, c, comps, meta_block, coarse_block = fut.result()
                clusters.extend(c)
                total_comparisons += comps
                metadata_blocked += meta_block
                coarse_blocked += coarse_block
                if coord is not None and c:
                    coord.add_near_dupe_clusters([list(s[0]) for s in c])
            except Exception as e:
                logger.error(f"Phase B album '{album}' failed: {e}")
                html_lines.append(f"Error scanning album '{album}'")

    html_lines.append("</pre>")
    html_str = "\n".join(html_lines)
    if coord is not None:
        coord.set_html_section("B", html_str)

    cross_clusters: List[Tuple[Set[str], float]] = []
    if enable_cross_album:
        cross_lines = ["<h2>Phase C – Cross-Album Near-Duplicates</h2>", "<pre>"]
        by_song: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for p in paths:
            key = (
                file_infos[p].get("primary") or "",
                file_infos[p].get("title") or "",
            )
            by_song[key].append(p)

        song_items = sorted(by_song.items(), key=lambda x: (x[0][0], x[0][1]))
        total_songs = len(song_items)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for idx, ((prim, title), song_paths) in enumerate(song_items, start=1):
                if len(song_paths) < 2:
                    continue
                logger.debug(
                    f"Phase C: [{idx}/{total_songs}] Scanning '{title}' ({len(song_paths)} tracks)"
                )
                cross_lines.append(f"{title or '<no title>'} ({len(song_paths)} tracks)")
                futures[ex.submit(_scan_paths, song_paths, file_infos, threshold, mixed_codec_boost, log_callback)] = (prim, title)

            for fut in as_completed(futures):
                _key = futures[fut]
                try:
                    c, comps, meta_block, coarse_block = fut.result()
                    cross_clusters.extend(c)
                    if coord is not None and c:
                        coord.add_near_dupe_clusters([list(s[0]) for s in c])
                    total_comparisons += comps
                    metadata_blocked += meta_block
                    coarse_blocked += coarse_block
                except Exception as e:
                    logger.error(f"Phase C group '{_key}' failed: {e}")
                    cross_lines.append(f"Error scanning group '{_key}'")

        cross_lines.append("</pre>")
        cross_html = "\n".join(cross_lines)
        if coord is not None:
            coord.set_html_section("C", cross_html)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Phase B summary: {total_albums} albums scanned, {total_comparisons} comparisons, {len(clusters)} clusters"
        )

    log_callback(
        "   • Near-duplicate gating summary: "
        f"{metadata_blocked} metadata comparisons blocked, {coarse_blocked} coarse fingerprint comparisons blocked "
        "(duration/tempo normalization deferred)."
    )

    all_clusters = clusters + cross_clusters

    result = NearDuplicateResult(auto_deletes={}, review_groups=[])
    for cluster, max_distance in all_clusters:
        by_alb: Dict[str, List[str]] = defaultdict(list)
        for path in cluster:
            album = file_infos[path].get("album") or ""
            by_alb[album].append(path)
        for album_paths in by_alb.values():
            if len(album_paths) <= 1:
                continue
            scored = sorted(
                album_paths,
                key=lambda p: _keep_score(p, file_infos[p], ext_priority),
                reverse=True,
            )
            keep = scored[0]
            losers = scored[1:]
            if not losers:
                continue
            reason = (
                f"Near-duplicate of {os.path.basename(keep)} "
                f"(max fingerprint distance {max_distance:.3f}); review required"
            )
            result.review_groups.append(
                NearDuplicateGroup(
                    winner=keep,
                    losers=losers,
                    max_distance=max_distance,
                    reason=reason,
                )
            )
    return result
