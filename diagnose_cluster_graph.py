#!/usr/bin/env python3
"""Diagnostic tool for cluster graph generation and 3D visualization.

Run with:
    python diagnose_cluster_graph.py <library_path>

This checks:
1. Whether cluster_info.json exists and is valid
2. Data structure (X_3d, labels, tracks)
3. HTML generation success
4. Browser launching capability
"""
import sys
import os
import json
from pathlib import Path


def diagnose(library_path: str) -> None:
    """Run diagnostic checks on cluster graph setup."""
    lib_path = Path(library_path).resolve()
    docs_path = lib_path / "Docs"
    info_path = docs_path / "cluster_info.json"
    html_path = docs_path / "cluster_graph.html"

    print("=" * 70)
    print("AlphaDEX Cluster Graph Diagnostic Tool")
    print("=" * 70)
    print()

    # Check 1: Library path exists
    print(f"1. Library path: {lib_path}")
    if lib_path.is_dir():
        print("   ✓ Library directory exists")
    else:
        print("   ✗ Library directory NOT found")
        return

    # Check 2: Docs folder exists
    print(f"\n2. Docs folder: {docs_path}")
    if docs_path.is_dir():
        print("   ✓ Docs directory exists")
    else:
        print("   ✗ Docs directory NOT found")
        docs_path.mkdir(parents=True, exist_ok=True)
        print("   → Created Docs directory")

    # Check 3: cluster_info.json exists and is readable
    print(f"\n3. Cluster info file: {info_path}")
    if not info_path.is_file():
        print("   ✗ cluster_info.json NOT found")
        print("   → Run Clustered Playlists from the Clustered tab first")
        return

    print("   ✓ cluster_info.json exists")

    # Check 4: Parse JSON
    try:
        with open(info_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        print("   ✓ JSON is valid")
    except json.JSONDecodeError as e:
        print(f"   ✗ JSON parsing failed: {e}")
        return
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
        return

    # Check 5: Data structure validation
    print(f"\n4. Data structure validation")
    required_keys = {"X_3d", "labels", "tracks"}
    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        print(f"   ✗ Missing required keys: {missing_keys}")
        print(f"   → Available keys: {list(data.keys())}")
        return

    print(f"   ✓ All required keys present")

    # Check 6: Data dimensions
    n_points_3d = len(data.get("X_3d", []))
    n_labels = len(data.get("labels", []))
    n_tracks = len(data.get("tracks", []))

    print(f"\n5. Data dimensions")
    print(f"   - X_3d points: {n_points_3d}")
    print(f"   - Labels: {n_labels}")
    print(f"   - Tracks: {n_tracks}")

    if n_points_3d == 0:
        print("   ✗ No 3D points found")
        return

    if n_points_3d != n_labels:
        print(
            f"   ✗ Mismatch: X_3d has {n_points_3d} but labels has {n_labels}"
        )
        return

    if n_points_3d != n_tracks:
        print(
            f"   ✗ Mismatch: X_3d has {n_points_3d} but tracks has {n_tracks}"
        )
        return

    print("   ✓ All dimensions consistent")

    # Check 7: Cluster info
    clusters = set(l for l in data.get("labels", []) if l >= 0)
    noise_count = sum(1 for l in data.get("labels", []) if l < 0)
    print(f"\n6. Clustering summary")
    print(f"   - Clusters: {len(clusters)}")
    print(f"   - Noise points: {noise_count}")
    print(f"   - Cluster IDs: {sorted(clusters)}")

    # Check 8: Sample data points
    print(f"\n7. Sample data (first 3 points)")
    for i in range(min(3, n_points_3d)):
        x, y, z = data["X_3d"][i]
        lbl = data["labels"][i]
        track = data["tracks"][i]
        print(f"   [{i}] Cluster {lbl:2d} | ({x:7.2f}, {y:7.2f}, {z:7.2f}) | {track}")

    # Check 9: HTML generation
    print(f"\n8. HTML visualization file: {html_path}")
    if html_path.is_file():
        size_kb = html_path.stat().st_size / 1024
        print(f"   ✓ cluster_graph.html exists ({size_kb:.1f} KB)")
    else:
        print("   ✗ cluster_graph.html NOT found")
        print("   → Attempting to generate...")

        try:
            from cluster_graph_3d import generate_cluster_graph_html

            generate_cluster_graph_html(str(lib_path), log_callback=print)
            if html_path.is_file():
                size_kb = html_path.stat().st_size / 1024
                print(f"   ✓ Generated ({size_kb:.1f} KB)")
            else:
                print("   ✗ Generation failed")
        except Exception as e:
            print(f"   ✗ Generation error: {e}")

    # Check 10: HTML content validation
    if html_path.is_file():
        print(f"\n9. HTML content validation")
        with open(html_path, "r", encoding="utf-8") as fh:
            html_content = fh.read()

        checks = [
            ("DOCTYPE", "<!DOCTYPE html>" in html_content),
            ("Three.js CDN", "three.min.js" in html_content),
            ("Scene container", "scene-container" in html_content),
            ("Data embedded", '"X_3d"' in html_content),
        ]

        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {check_name}")

    # Final recommendations
    print(f"\n10. Recommendations")
    if html_path.is_file() and n_points_3d > 0:
        print(f"   ✓ Everything looks good!")
        print(f"   → Try opening in a modern browser (Chrome, Firefox, Safari)")
        print(f"   → File: {html_path}")
        print(f"   → URL: file://{html_path}")
        print(f"   → Press F12 in browser to check console for errors")
    else:
        print(f"   → Ensure cluster_info.json has valid data")
        print(f"   → Run Clustered Playlists to regenerate cluster_info.json")
        print(f"   → Check browser console (F12) for JavaScript errors")

    print()
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_cluster_graph.py <library_path>")
        print()
        print("Example:")
        print("  python diagnose_cluster_graph.py ~/Music")
        sys.exit(1)

    diagnose(sys.argv[1])
