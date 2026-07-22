# Report: Analysis of Jiggle/Shaking Motion During Card Resizing

## Root Cause of the Jiggle
The shaking motion observed when the `LiquidGlassTile` resizes is a classic artifact of mixing **fractional scaling** with **integer-based window/layout geometry** within a dynamic animation loop. Specifically, the following factors contribute to the issue:

1. **Integer Truncation in Layout Bounds:**
   During each frame (`_tick`), the `ScalableTileWrapper` calculates its new fixed size by multiplying its internal bounding rect by the `_current_scale`. However, Qt widgets and layouts operate on integer pixels. When we compute `int(math.ceil(w)) + 2`, the resulting pixel dimension snaps to whole numbers. As the fractional scale smoothly interpolates, the wrapper size "stair-steps", causing a 1px to 2px jitter on the edges every few frames.
   
2. **Layout Recalculation Overhead (Reflow):**
   Because the wrapper's size is constantly mutating, the parent `QHBoxLayout` and the main window's `QScrollArea` are repeatedly forced to recalculate their layouts. The parent containers attempt to align the shifting integers to the layout grid, compounding the visual shaking effect.
   
3. **QPainter Aliasing during Proxy Transformation:**
   When scaling a `QGraphicsProxyWidget`, `QGraphicsView` transforms the painted buffer. If the translation matrix doesn't perfectly align to pixel boundaries (sub-pixel misalignment), horizontal and vertical lines (like text baselines and the inner glass borders) antialias differently from frame to frame, creating a shimmering or vibrating optical illusion known as pixel-snapping.

## Proposed Solutions for Smooth Transitions

To resolve the shaking motion and achieve a completely smooth, high-end transition, consider one of the following architectural changes:

### Solution 1: Fixed-Container Scaling (Recommended)
Instead of forcing the wrapper widget to change its `fixedSize` every 16ms, give the wrapper a generous, static size that accommodates the maximum possible scale (e.g., scale level 5). 
- Animate the `_current_scale` purely internally within the `QGraphicsView` by updating the transform, but **leave the view's physical widget size unchanged**.
- Because the layout isn't shifting, the parent window doesn't reflow, completely eliminating structural jitter. The card will appear to grow smoothly inside a fixed invisible bounding box.

### Solution 2: QPropertyAnimation & Subpixel Precision
Replace the manual `_tick` loop with a native `QVariantAnimation` animating a custom scale property.
- Combine this with a custom `paintEvent` directly inside the `LiquidGlassTile` instead of using `QGraphicsView`. 
- Use `QPainter.scale(scale, scale)` to draw the widget natively at float sizes. This is mathematically smoother as `QPainter` has better subpixel antialiasing control than `QGraphicsProxyWidget`, preventing the aliasing shimmer.

### Solution 3: Hardware-Accelerated Composition (OpenGL / QtQuick)
If this is intended for an overkill, "art piece" UI, rendering the widget to an offscreen buffer (FBO) and composing it using a `QOpenGLWidget` allows for perfect GPU interpolation. The card would be rendered once at high resolution, and the GPU would handle the scaling transformation, guaranteeing 0 reflow overhead and sub-pixel perfect scaling.