# Progressive Loading Design — Splash Screen to Landing

## Overview

The splash screen no longer displays a static progress animation. Instead, it now shows **real, actionable progress** that reflects actual asset loading:

1. **Phase 1 (0-50%)** — Fast initial load (750ms) while UI components initialize
2. **Phase 2 (50-100%)** — Monitored image loading; bar advances as real images load from the library
3. **Timeout Safety** — If images take longer than 5 seconds, continues automatically

## User Experience

### Before
```
[████████████████████████████]
User sees: Constant animation for 1500ms, then splash fades
Reality: Images may still be loading in background; no feedback
```

### After
```
Phase 1 (UI loading):     [██████████          ] 50% (750ms)
Phase 2 (images loading):  [██████████████████] 100% (images arrive)
Actual progress:           Bar advances as images load, not elapsed time
Timeout:                   If >5s with no images, continues anyway
```

## Code Architecture

### Three Components

#### 1. **SplashScreen** (`gui/widgets/splash.py`)

**New Constants:**
```python
_FILL_PHASE1_MS = 750         # Time to reach 50%
_FILL_PHASE2_MS = 750         # Time from 50% to 100%
_IMAGE_LOAD_TIMEOUT_MS = 5000 # Max wait for images
```

**New State:**
```python
self._phase2_started: bool     # Track if we're in image-loading phase
self._images_loaded: int       # Count of loaded images
self._images_target: int       # Expected total images
```

**New Animations:**
```python
self._fill_phase1   # 0.0 → 0.5 (750ms, InOutCubic)
self._fill_phase2   # 0.5 → 1.0 (750ms, InOutCubic)
```

**New Methods:**

```python
def set_image_target(count: int) -> None:
    """Set expected image count for progress mapping."""
    self._images_target = count

def report_image_loaded(index: int, _image=None) -> None:
    """Called by art scanner; updates progress bar with real image count."""
    # Maps loaded image count to 50%-100% progress range
```

**New Slot:**

```python
def _on_phase1_done() -> None:
    """Start 5-second timeout waiting for images, or continue if timeout."""
    # Sets up QTimer to eventually call _continue_to_phase2()

def _continue_to_phase2() -> None:
    """Start phase 2 animation (50% → 100%)."""
    # Called either when images arrive OR after 5-second timeout
```

#### 2. **MosaicLanding** (`gui/widgets/landing.py`)

**New Public Method:**

```python
def wire_splash_progress(splash) -> None:
    """Connect this landing's image loading to splash progress.

    - Sets splash's image target to number of tiles
    - Connects _ArtScanner.art_found signal to splash.report_image_loaded()
    """
```

This method:
- Tells splash how many images to expect
- Wires the scanner's `art_found` signal directly to the splash
- Allows real-time progress feedback without tight coupling

#### 3. **Main Orchestration** (`alpha_dex_gui.py`)

**New Wiring:**

```python
landing = MosaicLanding(shared_geo, saved_lib)

# Wire splash to landing for real progress
landing.wire_splash_progress(splash)
```

This single call:
- Connects the splash and landing
- Enables real-time progress reporting
- Maintains clean separation of concerns

## Timeline & State Flow

### 1. Splash Shows (t=0ms)
```
SplashScreen.__init__()
├─ _fill_phase1.start()  // 0 → 0.5 over 750ms
└─ _progress = 0.0
```

### 2. Phase 1 Completes (t=750ms)
```
_fill_phase1.finished → _on_phase1_done()
├─ Create 5-second timeout
├─ Start waiting for images
└─ _phase2_started = False (still in Phase 1)
```

### 3. Image #1 Arrives (t=850ms, example)
```
_ArtScanner.art_found(index=0, image=...)
│
└─ Lands on splash.report_image_loaded(0, ...)
   ├─ self._images_loaded = 1
   └─ If phase2 has started:
      └─ _progress = 0.5 + (1/34 * 0.5) ≈ 0.515
         (Maps 1 of 34 images to ~51.5% bar progress)
```

### 4. Phase 2 Starts (t≥750ms, triggered by either):

#### Option A: Images Arrive Quickly
```
Many images arrive (e.g., images 2-34)
│
└─ splash.report_image_loaded() called repeatedly
   ├─ _images_loaded increments
   ├─ _progress updates (0.5 → 1.0)
   └─ [At image #34, timeout still running]
```

#### Option B: Timeout Expires (t=5750ms)
```
_image_load_timeout.timeout() → _continue_to_phase2()
├─ If few/no images loaded yet
└─ Start _fill_phase2 (0.5 → 1.0 over 750ms)
```

### 5. Phase 2 Completes (t=750ms + 750ms = 1500ms, or after images)
```
_fill_phase2.finished → _start_fade()
├─ reveal_ready.emit()  // Tell main window to show
└─ _fade_anim.start()   // 450ms fade out
```

## Edge Cases Handled

### No Library (First Launch)
```
landing.wire_splash_progress(splash)
├─ _ArtScanner not created (no saved_path)
└─ splash waits 5 seconds, then continues
   ├─ No images arrive
   ├─ Timeout triggers _continue_to_phase2()
   └─ Bar continues to 100% normally
```

### Fast Image Loading
```
Images arrive before Phase 2 starts
├─ report_image_loaded() calls happen
├─ _phase2_started is still False
└─ Progress updates are ignored
   (They'll be applied once _continue_to_phase2() runs)
```

### Timeout + Images Arriving
```
Some images before 5s, more after
├─ Early images trigger phase 2 start (OR timeout does)
├─ Bar jumps to 50%
├─ More images arrive → bar advances 50% → 100%
└─ OR timeout expires → bar animates 50% → 100%
```

### Very Slow I/O (>5s)
```
t=5000ms: No images yet
├─ Timeout fires → _continue_to_phase2()
├─ Bar starts animating 50% → 100% (750ms)
├─ Meanwhile, images still arriving
└─ Bar updates reflected in phase 2 animation
```

## Progress Bar Calculation

### Phase 1 (0-50%)
```
progress = animation_value  // 0.0 → 0.5 over 750ms
```

### Phase 2 (50-100%)
```
if _images_target > 0:
    image_fraction = min(1.0, _images_loaded / _images_target)
    progress = 0.5 + (image_fraction * 0.5)

Examples:
  0 images of 34   → 0.5 + 0.0 = 50.0%
  1 images of 34   → 0.5 + 0.015 = 51.5%
  17 images of 34  → 0.5 + 0.25 = 75.0%
  34 images of 34  → 0.5 + 0.5 = 100.0%
```

## Key Features

### ✓ Real Progress Feedback
- User sees actual image loading, not just time passing
- Bar doesn't reach 100% until real work is done (or timeout)

### ✓ Timeout Safety
- 5-second maximum wait prevents indefinite blocking
- Continues automatically if library has no images or I/O is slow

### ✓ Clean Architecture
- Splash knows nothing about landing internals
- Landing wires its scanner via `wire_splash_progress()`
- Main code calls one method to connect them

### ✓ Thread-Safe
- `art_found` signal is thread-safe (emits from scanner thread)
- `report_image_loaded()` updates from main thread via signal
- No locks needed; Qt signal delivery handles synchronization

### ✓ Backward Compatible
- If no splash is wired, landing works normally
- If images arrive very slowly, timeout ensures completion
- Existing UI flow unchanged

## Testing Checklist

- [ ] **Normal case** (good I/O): Bar reaches 50%, then advances as images load
- [ ] **No library** (first launch): Times out, continues to 100%
- [ ] **Slow I/O** (>5s): Timeout triggers, bar animates anyway
- [ ] **Fast I/O** (<1s): Bar visible at 50%, then jumps to 100% as images arrive
- [ ] **Interrupted** (user closes during splash): Cleanup happens gracefully
- [ ] **Multiple launches** (resume same library): Second time loads faster
- [ ] **Network paths**: Slow I/O on network-mounted libraries triggers timeout

## Visual Indicators

### Progress Bar States

```
┌─────────────────────────────────┐
│ AlphaDEX                        │
│ Music Library Manager · v2.0    │
│                                 │
│ ├─────────────────────────────┤ ← 0% (Before splash shows)
│                                 │
│ ├──────────────────────────────┤ ← 50% (Phase 1 done, waiting)
│                                 │
│ ├────────────────────────────┤ ← 75% (Some images loaded)
│                                 │
│ ├─────────────────────────────┤ ← 100% (All images loaded)
│                                 │
└─────────────────────────────────┘
```

## Performance Impact

- **Minimal overhead**: `report_image_loaded()` is O(1) — just math and redraw
- **No blocking**: Doesn't wait for I/O (uses timeout)
- **Clean shutdown**: Timeout cancelled automatically when phase 2 starts
- **Memory efficient**: Only stores counts, not image data

## Future Enhancements

- [ ] Add visible image count text ("Loading 12 of 34 images...")
- [ ] Support partial image loading (show progress for subsets)
- [ ] Per-phase timing configuration
- [ ] Detailed logging of phase transitions
- [ ] Option to skip loading bar entirely for very fast systems
