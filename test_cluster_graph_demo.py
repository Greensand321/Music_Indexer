#!/usr/bin/env python3
"""Quick demo script to test the 3D cluster graph visualization.

Run with:
    python test_cluster_graph_demo.py

This generates a demo HTML file with 10 random points in 3D space
and opens it in your default browser. Useful for diagnosing issues
with the Three.js visualization pipeline.
"""
import random
import os
import sys
import webbrowser
from pathlib import Path

# Add repo to path
sys.path.insert(0, os.path.dirname(__file__))

from cluster_graph_3d import generate_cluster_graph_html_from_data


def main():
    """Generate and open demo graph."""
    output_dir = Path(__file__).parent / "Docs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "cluster_graph_demo.html"

    # Create demo data: 10 random points in 3D space, 2 clusters
    random.seed(42)
    n_points = 10
    demo_data = {
        "X_3d": [
            [random.uniform(-20, 20) for _ in range(3)]
            for _ in range(n_points)
        ],
        "labels": [i % 2 for i in range(n_points)],
        "tracks": [f"demo_track_{i}.mp3" for i in range(n_points)],
        "metadata": [
            {
                "title": f"Demo Track {i}",
                "artist": "Test Artist",
                "duration": 180 + i * 10,
                "bpm": 120,
            }
            for i in range(n_points)
        ],
        "cluster_info": {
            "0": {"size": 5, "genres": [], "tempo": [100, 140]},
            "1": {"size": 5, "genres": [], "tempo": [100, 140]},
        },
        "X_downsampled": False,
        "X_total_points": n_points,
    }

    print(f"Generating demo 3D graph with {n_points} random points...")
    print(f"Output: {output_file}")

    try:
        path = generate_cluster_graph_html_from_data(
            demo_data, str(output_file), log_callback=print
        )
        print(f"✓ Generated: {path}")
        print()
        print("Opening in browser...")
        webbrowser.open(Path(path).as_uri())
        print("✓ Done!")
        print()
        print("Troubleshooting tips:")
        print("- If nothing appears: check browser console (F12) for JS errors")
        print("- Points should be colored by cluster (hue rotation)")
        print("- Drag to orbit, scroll to zoom, right-drag to pan")
        print("- Click points to select them")
        print(f"- HTML file at: {output_file}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
