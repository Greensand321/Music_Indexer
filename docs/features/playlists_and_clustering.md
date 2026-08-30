# Feature — Playlists, Clustering & the Visual Music Graph

*This is the analytical heart of AlphaDEX: the part that doesn't just organize your
music but listens to it and finds structure in it. It spans three connected ideas —
rule-based playlists, machine-learning clustering, and a visual map of your
collection. This document explains all three in plain language, with the reasoning and
trade-offs behind each, and is honest about what is built today versus still planned.*

---

## Three levels of "make me a playlist"

There are three increasingly clever ways AlphaDEX can build playlists, and they form a
natural ladder:

1. **By rules you understand** — "give me all my fast, high-energy tracks." Predictable
   and transparent.
2. **By learned similarity** — "group my library into clusters of songs that *sound*
   alike, even if I can't say why." Powerful and surprising.
3. **By visual exploration** — "show me my whole collection as a map, and let me lasso
   a region to turn it into a playlist." Hands-on and exploratory.

Each level builds on the one before. We'll take them in order.

## Level 1: Rule-based playlists

The simplest mode generates ordinary **playlist files** (`.m3u` lists that any music
player understands) by sorting your tracks according to rules you choose. The
available dimensions are intuitive:

- **Tempo** — how fast the track is, in beats per minute, bucketed into slow / medium
  / fast.
- **Energy** — how loud and intense the track feels, bucketed into low / medium /
  high. (Energy here is measured from the loudness of the audio, not from any tag.)
- **Folder, genre, and year** — straightforward groupings by where a track lives, what
  it's labeled, or when it came out.

You can even combine dimensions — "slow but high-energy," for instance — to carve out
specific moods.

### Auto-DJ: playlists that flow

A standout of this level is **Auto-DJ chaining**. Instead of just collecting matching
tracks, Auto-DJ *orders* them so that each song flows naturally into the next. The
metaphor is a thoughtful DJ who would never slam from a 70-BPM ballad straight into a
140-BPM banger. Starting from a seed track, the app repeatedly looks for the remaining
track that is most similar in feel to the one just played, and chains it on — building
a set where the transitions feel smooth rather than jarring. Under the hood it judges
"similar in feel" using a compact sound-summary of each track, but you don't need to
think about that; you just hear a set that hangs together.

### Auto-DJ, in a bit more technical detail

Under the hood, Auto-DJ's "similar in feel" judgment reuses the exact same
27-dimensional sound-summary (mean and spread of the track's timbre, plus its tempo)
that the clustering engine below is built on — so a track's position for Auto-DJ
purposes and its position for clustering purposes are the same measurement. The
chaining algorithm itself is a simple, greedy "always pick the closest next track"
walk: starting from your seed track, it repeatedly picks whichever *remaining* track
is closest (by plain distance in that 27-dimensional space) to the track it just
added, then moves on from there. It has no lookahead and doesn't try to avoid
"painting itself into a corner" late in a long playlist — it's a nearest-neighbor
tour, not an optimized route. One detail worth knowing if a queued set feels uneven:
this distance calculation does not currently apply the same feature-scaling step the
clustering wizard uses (see the "honest note on features" below), so a feature with
naturally larger numbers can end up dominating the "how similar is this" judgment
more than a listener's ear might expect.

### A smaller helper: genre normalization

**Genre normalization** tidies the messy, inconsistent genre labels in your files by
looking each track up in an online music encyclopedia and writing back a small set of
tags — so your library moves toward fewer stray spellings of "Hip-Hop." Worth being
precise about what this does today, though: the Playlist Creator side of the app
performs a straightforward "ask MusicBrainz, take its three most popular tags, write
them if the file doesn't already have at least two genres" pass — it does not map
messy variants to one clean, chosen vocabulary. A genuinely canonical genre-mapping
system does exist in the codebase, but only in the legacy Tkinter app; see
**features/tag_fixer.md** for the full explanation of why "genre normalizer" means
two different things depending on which app you're in.

*(An earlier version of this document also described a "year-gap assistant" that
built playlists telling a chronological story across your collection. That feature
does not exist. The closest thing in the codebase is a differently-purposed "Year
Assistant," legacy-app-only, that fills in *missing* year tags via a copy-paste-to-an-LLM
workflow — it has no connection to playlist ordering. If chronological playlist
pacing is something you want, it's a genuine gap, not a documentation oversight; see
**ROADMAP.md**.)*

## Level 2: Clustering — letting the music sort itself

This is where AlphaDEX gets genuinely clever. **Clustering** means handing your
library to a mathematical process that **groups tracks by how similar they sound**,
without anyone telling it what the groups should be. The analogy is letting an
extraordinarily musical friend loose on your collection and saying "sort these into
piles that belong together" — and watching them produce piles you'd never have thought
to make, cutting across genre labels based on the actual texture and feel of the music.

### How can a computer judge "sounds alike"?

It can't hear the way you do, so it measures. For each track, AlphaDEX extracts a set
of **audio features** — numeric descriptions of measurable qualities of the sound — and
treats them as the track's coordinates. The most important feature in use today is
**timbre**: the "tone color" of a track, the quality that makes a saxophone and a
guitar playing the same note sound different. Timbre is captured as a cluster of
related numbers describing the shape of the sound. Alongside timbre, the app uses
**tempo** (speed). Together these give each track a position in a multi-dimensional
"space of sound," where tracks that sound alike sit near each other and tracks that
sound different sit far apart.

> **An honest note on features.** The interface offers checkboxes for additional
> qualities — harmonic content, brightness, percussive density — and these are part of
> the design's vision. Today, however, the clustering actually computes **timbre and
> tempo**; the other checkboxes are scaffolding for features not yet wired into the
> engine. **ROADMAP.md** tracks the work to bring the rest online. We mention this so
> the documentation matches reality rather than the aspiration.

### Two ways to form the groups

AlphaDEX offers two different grouping strategies, and the difference between them is
worth understanding because it changes the results:

- **K-Means** is the "divide the room into a fixed number of circles" approach. You
  tell it how many groups you want — say, eight — and it partitions every track into
  exactly that many groups, each track belonging to one. It is fast, predictable, and
  ideal when you already have a target in mind ("I want eight mood playlists"). Its
  weakness is that it will always produce exactly that many groups whether or not the
  music naturally falls into that many, and you have to choose the number yourself.

- **HDBSCAN** is the "find the natural crowds" approach. You don't tell it how many
  groups to make; it discovers them, identifying dense clusters of similar tracks and
  setting genuinely odd one-off tracks aside as "outliers" rather than forcing them
  into a group where they don't belong. It is more faithful to the real shape of your
  collection, at the cost of being slower and less predictable — on a small library it
  may produce many tiny groups.

Neither is "better"; they answer slightly different questions. K-Means is a confident
sorter following your instructions; HDBSCAN is a naturalist cataloguing what's actually
there.

### The wizard: configuring without expertise

Because these choices could overwhelm a newcomer, the enhanced clustering room wraps
them in a **step-by-step wizard** that asks plain questions in sequence:

1. **Which qualities of the sound should I pay attention to?** (feature selection,
   with friendly presets like "Fast," "Balanced," and "Complete")
2. **How should I put the different qualities on an equal footing?** (normalization —
   making sure one feature with naturally large numbers doesn't drown out the others)
3. **Which grouping strategy, and with what settings?** (K-Means or HDBSCAN)
4. **What should I do with tiny or stray groups?** (clean-up — merge them into a
   "Miscellaneous" pile, or set them aside)
5. **What should I produce?** (playlists, a quality report, a visual graph)

The wizard's value is that it lets you express intent — "group my library into eight
piles based mainly on rhythm and texture" — without needing to know the mathematics
underneath. The app translates your answers into the technical settings.

### Was it a *good* grouping? Quality scores

Clustering always produces *some* answer, but not every answer is meaningful. To help
you judge, AlphaDEX computes **quality scores** for a grouping. The most intuitive is
the **Silhouette score**, which measures whether each track sits comfortably inside its
own group (close to its group-mates) versus uncomfortably near a neighboring group. A
high score means the groups are clean and well-separated; a low score means they're
muddy and overlapping — a hint that you should try a different number of groups or
different features. (Two companion scores measure related notions of how distinct and
compact the groups are.) Think of these as a report card on the grouping, telling you
whether to trust it or try again.

### From groups to playlists

Finally, each discovered group becomes a playlist. The tracks HDBSCAN judged to be
true outliers can either be gathered into their own "Miscellaneous" list or left out
entirely, so your clean mood-playlists aren't polluted by the one weird track that
fits nowhere.

## Level 3: The Visual Music Graph

Numbers and playlists are useful, but the most striking way to understand your
collection is to **see** it. The Visual Music Graph turns your library into an
interactive picture: a scatter of dots where **each dot is a track**, and dots that
sit near each other represent tracks that sound alike. Whole regions of the picture
become visible "neighborhoods" of similar music. You can zoom and pan around it, hover
over a dot to see what track it is, click to inspect, and — most powerfully — **draw a
selection around any cluster of dots to turn that region directly into a playlist.**
It's the difference between reading a list of your music and walking through a map of
it.

### How do you draw a 13-dimensional space on a flat screen?

Here is the conceptual hurdle, and the elegant trick that solves it. Your tracks live
in a space with many dimensions — one axis for every audio feature. A space like that
is impossible to picture directly; nobody can visualize thirteen perpendicular
directions at once.

The solution is **dimensionality reduction**: a mathematical way of **flattening** that
many-dimensional space down to the two dimensions of your screen while doing its best
to **preserve the relationships** — keeping tracks that were close together still close,
and tracks that were far apart still far. The analogy is casting the shadow of an
intricate 3-D sculpture onto a flat wall: you lose some depth, but the overall shape and
the way the parts relate are still recognizable. AlphaDEX uses well-established methods
for this flattening (favoring a fast modern technique when it's available, and falling
back to a slower but vivid alternative otherwise), so the map you see is a faithful
two-dimensional shadow of the true high-dimensional arrangement of your music.

### What actually opens today: a 3-D browser view

The "Music Graph" room in the sidebar is, today, a **launcher**: it checks whether
cluster data is available, and when you click through, it opens a standalone web page
in your default browser — a Three.js/WebGL 3-D scatter plot you can orbit, generated
fresh from your most recent clustering run.

Beyond orbiting and hovering, that page gives you: **spread control** (a slider plus
1× / 10× / 30× / 50× / 100× presets) to pull the clusters apart until the structure
is readable; **axes, grid, and orbit-ring toggles** for spatial reference; a
click-to-select flow that exports a selection as CSV or an `.m3u` playlist; a legend
that doubles as per-cluster show/hide switches; and an **Import JSON** button that
lets you drop a different cluster file onto the same viewer without regenerating it
from the app.

> **A note if you used this before and found it dead.** For some time the generated
> page was producing invalid JavaScript — the step that inlines your library's data
> into the page left a stray token behind, which is a syntax error, so the whole
> visualization silently failed to run. That's fixed. It went unnoticed for a while
> because the automated tests only checked that certain *text* appeared in the
> generated file, never that the file's code actually parsed; there's now a test that
> checks the latter specifically.

### The in-app 2-D map

The Music Graph workspace also draws the map **inside the app**, which is where the
hands-on work happens. The window is split between the plot itself and a side column
holding the cluster legend and a details panel.

Three interaction modes sit above the plot:

- **Pan** — drag to move around, the ordinary way to browse.
- **Rectangle** — drag a box; everything inside it is selected.
- **Lasso** — draw a freehand loop around an irregular region, which is usually what
  you actually want, since clusters aren't box-shaped.

Holding **Ctrl or Shift** while selecting *adds* to the current selection rather than
replacing it, so you can gather several separate pockets of the map into one set.

Hovering a dot identifies it immediately. **Clicking** one goes further and reads the
file itself, filling in real artist / title / album / genre / year and its embedded
cover art — hover stays deliberately cheap (no disk access) because it fires
constantly, while a click is a considered act and can afford one read.

The **legend** doubles as controls: each row shows a cluster's colour and track
count, its checkbox hides or shows that cluster on the plot, and clicking the row
selects every track in it. Unclustered "noise" tracks are listed last and drawn in
grey, so they read as leftovers rather than as another cluster.

Once you have a selection, you can **send it straight to the Player** as a queue, or
export it as a **CSV** or an **`.m3u` playlist** — the point of the whole exercise:
see a region of your library that sounds alike, and turn it into something you can
listen to.

> **If PyQtGraph isn't installed**, the plot area explains that and tells you how to
> install it, and the 3-D browser view keeps working regardless.

The "Open Visual Graph" button in the clustering workspace's Results tab does now
take you straight to the Music Graph room (it used to just show a message box asking
you to click the sidebar yourself). Workspaces don't switch themselves — the button
raises a `navigate_requested` signal and the main window performs the move, keeping
navigation in one place.

## Two workspaces exist; only one is live

If you go looking at the code, you'll find two clustering workspaces:
`clustered.py`, an older, simpler room, and `clustered_enhanced.py`, the
three-tab (Quick Start / Advanced / Results) workspace this document has been
describing. Only the enhanced one is reachable — the sidebar's "Clustered" entry
is wired to it, and the older workspace is unreferenced dead code left in the
repository. If you're reading the source to understand this feature, ignore
`clustered.py`; it doesn't run.

*(Historical note on the quality report: the dialog that displays your Silhouette /
Davies-Bouldin / Calinski-Harabasz scores used to fail to render its per-cluster
breakdown against any real result — it called an internal method that didn't exist
under that name. That's fixed; the scoring math was never affected.)*

## What's real today, and what's still ahead

In the spirit of an honest, up-to-date resource:

**Working today:** rule-based playlists and Auto-DJ; MusicBrainz-based genre
tag-filling (see the caveat above on what "normalization" means here); clustering by
timbre and tempo with both K-Means and HDBSCAN; the five-step configuration wizard;
the quality scores and the report dialog that displays them; automatic playlist
creation from clusters; the 3-D browser visualization; and the in-app 2-D map with
pan/rectangle/lasso selection, hover and click inspection, legend-driven cluster
visibility, and selection → Player / CSV / `.m3u`.

**Still ahead (see ROADMAP.md):** bringing the additional sound features (harmonic
content, brightness, percussive density) into the actual clustering rather than just
the checkboxes; live re-tuning of a clustering without recomputing everything from
scratch; in-map cluster *editing* (merging clusters, moving a track between them);
selecting by distance from a point and filtering the map by metadata; a suggestion
engine for weak groupings; and a one-click way to export the quality report.

---

*This concludes the feature walkthroughs. For everything that is planned but not yet
finished across the whole application, see **ROADMAP.md**.*
