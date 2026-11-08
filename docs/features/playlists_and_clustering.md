# Feature — Playlists, Clustering & the Sound Map

*This is the analytical heart of AlphaDEX: the part that doesn't just organize your
music but listens to it and finds structure in it. It spans three connected ideas —
rule-based playlists, machine-learning clustering, and a visual map of your
collection. This document explains all three in plain language, with the reasoning and
trade-offs behind each, and is honest about what is built today versus still planned.*

---

**Where to find it.** All three live in the sidebar's **Listen** group. Rule-based
playlists are the **Playlists** page (headed "Playlist Generator"). Clustering and the
map share one page, the **Sound Map**, with two sections: **Sound Map → Analyse** runs
the clustering, and **Sound Map → Map** shows the result. (Older material calls these
the "Clustered" and "Music Graph" pages, or the "Visual Music Graph".)

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
- **Folder** — one playlist per album folder. (Genre playlists exist in the engine,
  `playlist_engine.export_genre_playlists`, but the modern app doesn't offer them yet,
  and there is no year grouping; see **ROADMAP.md**.)

You can even combine dimensions — "slow but high-energy," for instance — to carve out
specific moods. The tempo and energy *range* fields narrow this further: tracks
outside the chosen BPM / energy window are left out of every bucket.

**Nothing is written without your say-so.** Every mode of the Playlists page
(folder, tempo/energy, Auto-DJ and repair) first works out exactly which `.m3u`
files it would write, then shows you that list — marking which existing playlists
would be *replaced* — and only writes after you confirm. Any playlist it replaces is
copied to `name.m3u.<date-time>.bak` beside it first, and every write and backup is
appended to `Docs/playlist_writes.log`. A Cancel button stops a long analysis.

- **Folder playlists** are one per *album folder* (every folder that directly holds
  audio files), named after its path (`Artist_Album.m3u`). The library's own
  folders — Docs, Playlists, Trash, Quarantine, Not Sorted, Manual Review — are
  skipped.
- **Auto-DJ** needs a seed track from your library; it no longer silently starts
  from whichever file happens to be found first. A track whose audio can't be
  analysed (a corrupt or undecodable file) is skipped and named in the log —
  it used to fail the whole run — and only an unreadable seed is an error.
  Cancel now works while the library is being analysed, not just afterwards.
- **Playlist repair** checks the playlists you add (or, with an empty list, every
  playlist in the library's `Playlists/` folder). A missing entry is first pointed at
  the same track in the other codec (FLAC → Opus when "Prefer Opus" is ticked, Opus →
  FLAC otherwise), and otherwise matched by file name across the library. You see the
  fixes, and the entries that are still missing, before anything is saved.

The analysis caches (`Docs/tempo_energy.npy`, `Docs/features*.npy`) are stored in a
plain numeric format and are never unpickled; a corrupt cache — or one written by an
older version — is simply rebuilt, so the first run after upgrading re-analyses once.
A track whose tempo or energy comes out as "not a number" is treated as a failed
analysis: it is left out of every bucket (it used to land in fast/high) and is not
cached. A track listed twice is bucketed once.

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
tour, not an optimized route. Every track is used at most once (the same file listed
twice is one track), ties go to the track found first, so the same seed and library
always give the same set, and the walk always ends. Each run analyses the whole
library again (the app does not yet hand Auto-DJ a saved feature cache), which is the
slow part on a large library. One detail worth knowing if a queued set feels uneven:
this distance calculation does not currently apply the same feature-scaling step the
clustering wizard uses, so a feature with
naturally larger numbers can end up dominating the "how similar is this" judgment
more than a listener's ear might expect.

### A smaller helper: genre normalization

**Genre normalization** tidies the messy, inconsistent genre labels in your files so
your library moves toward fewer stray spellings of "Hip-Hop" — which matters here
because clean genres make every genre-based view of the library (and any future
genre playlists) more useful. The **Metadata** page has two sections for this:
**Metadata → Fill genres** fills *missing* genres from MusicBrainz, and
**Metadata → Normalize genres** collapses whatever vocabulary your library has
accumulated into a canonical one via a mapping you build once, preview, and apply.
See **features/tag_fixer.md** for both.

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
treats them as the track's coordinates. Six qualities are available, and the wizard
lets you choose which ones to listen to:

- **Timbre** — the "tone color" of a track, the quality that makes a saxophone and a
  guitar playing the same note sound different. Captured as a cluster of related
  numbers describing the shape of the sound (26 of them). This is the workhorse.
- **Tempo** — speed, in beats per minute.
- **Harmonic content** — how much energy sits in each of the twelve pitch classes,
  which is a fingerprint of key and tonality (12 numbers).
- **Brightness** — where the sound's energy sits in the frequency range, and how much
  that moves around; bright, airy tracks versus dark, muffled ones.
- **Energy** — overall loudness and how much it varies.
- **Percussive density** — how many note onsets happen per second; busy, rhythmic
  material versus sparse, sustained material. (This one is the slowest to compute,
  which is why it's off by default.)

Only the qualities you tick are actually computed, so leaving an expensive one
unticked genuinely saves the work. Together the chosen qualities give each track a
position in a multi-dimensional
"space of sound," where tracks that sound alike sit near each other and tracks that
sound different sit far apart.

> **A note on caching.** Computing these qualities is the slow part of clustering,
> so the results are remembered per track between runs. Each *combination* of ticked
> qualities (and each extraction engine) keeps its own remembered set, because a
> vector built from timbre-and-tempo means something different from one built from
> timbre-and-harmony, even when they happen to be the same width. The practical
> upshot: the first run with a new combination re-analyzes your library once; every
> run after that with the same combination is fast, and switching back to an earlier
> combination is fast too.
>
> *(Historical note: for a long stretch only timbre and tempo were actually computed,
> and the other four checkboxes were scaffolding that the engine ignored. That's no
> longer the case — every checkbox now does what it says.)*

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

Because these choices could overwhelm a newcomer, **Sound Map → Analyse** offers two
ways in. Its **Quick Start** tab runs one fixed, sensible configuration (eight K-Means
groups from tempo, timbre and energy). Its **Advanced** tab wraps every choice in a
**step-by-step wizard** (**Configure with Wizard…**) that asks plain questions in
sequence:

1. **Which qualities of the sound should I pay attention to?** (feature selection,
   with friendly presets like "Fast," "Balanced," and "Complete")
2. **How should I put the different qualities on an equal footing?** (normalization —
   making sure one feature with naturally large numbers doesn't drown out the others)
3. **Which grouping strategy, and with what settings?** (K-Means or HDBSCAN)
4. **What should I do with tiny or stray groups?** (clean-up — *shown but not
   available yet*: every group the algorithm finds is kept)
5. **What will I get?** (the data for the map, the 3-D view and Save cluster
   playlists, plus a quality report). One option here is live: **Offer to save a
   playlist per cluster when the analysis finishes** (off by default) opens the Save
   cluster playlists preview after a successful run, and nothing is written until
   you click Save there. The report and auto-open options are shown but not
   available yet.

The dimensionality-reduction choice on step 2 is likewise shown disabled: clustering
always uses every selected feature, and the maps flatten them automatically for
display only. HDBSCAN needs the optional `hdbscan` package; K-Means (and Auto-DJ)
work without it.

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

The scores are computed on clustered tracks only: HDBSCAN's "unclustered" tracks
are not a group, so they are left out of the scores and reported separately (as is
the cluster count). Tracks whose audio could not be analysed are left out of the run
and listed, and are retried next time rather than being remembered as silence.

### From groups to playlists

The analysis itself writes no playlists. Each run saves its results in the
library's `Docs/` folder: `cluster_members.json` (which track is in which group, for
**every** track), `cluster_info.json` (what the map draws; above 5,000 tracks the
map shows an evenly spread 5,000 of them) and `cluster_graph.html`. Both JSON files
carry the same run identity, so the app can tell when they belong to different
analyses.

To turn the groups into playlists, use **Save cluster playlists…** at the top of the
Sound Map (it is there in both the Analyse and Map sections). It is preview-first,
like every other write in the app:

1. It first checks the analysis against your library (read-only, in the background,
   with a Cancel): which tracks are still there, whether the library has moved
   (an external drive back under another letter is mapped automatically), and how
   many tracks were added since. It also reads the tags of the most typical tracks
   of each group to name it.
2. The preview lists one row per group: its colour, a name you can edit, how many
   tracks it has, how many are missing, and whether Save would make a **New** file,
   **Replace** one, or **Skip** it (and why). Groups you hid in the map's legend
   start unticked. You choose the order inside each playlist (**closest to the centre
   first**, the most typical tracks first, or **library order**) and a minimum size
   (5 by default; smaller groups are skipped).
3. **Save N playlists** writes exactly what the preview shows. If any file would be
   replaced, you are asked first, and each replaced file is backed up beside it.
   Every write goes to `Docs/playlist_writes.log`.

Names are built from what the group's tracks have in common: the dominant genre (or
two, as `Techno & House`), otherwise a dominant artist, then the typical tempo and an
energy word — for example `Cluster 03 - Deep House - ~120 BPM - high energy`, or just
`Cluster 07` when nothing stands out. Names are made safe for Windows, macOS and
Linux, and kept short enough for Windows' path limit.

Each analysis gets its own folder, `Playlists/Clusters/<date time> <method>/`
(for example `2026-11-07 15.30.12 K-Means (k=8)`), so re-clustering never overwrites
an earlier set; saving the same analysis again replaces its own files (with backups)
and leaves any file you renamed alone. The playlists are ordinary `.m3u` files with
relative entries, so later Indexer moves and duplicate quarantines keep them working.

A few things to know: HDBSCAN's **unclustered** tracks are never saved as a
playlist. Analyses made before this feature have no membership file and need to be
run again (Save cluster playlists says so). Tracks added since the analysis are in no
group until you analyse again, and tracks retagged or re-encoded in place are not
noticed.

You can still turn any hand-picked part of the map into one playlist: click a
cluster in the legend (or lasso a region) and use **Export playlist…**. That export
now goes through the same writer, so a playlist it replaces is backed up and logged.

## Level 3: The Sound Map

Numbers and playlists are useful, but the most striking way to understand your
collection is to **see** it. **Sound Map → Map** turns your library into an
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

### The in-app 2-D map

The Map section draws the map **inside the app**, which is where the hands-on work
happens. The window is split between the plot itself and a side column holding the
cluster legend and a details panel. With no analysis yet, it says to run one first
(in the Analyse section).

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

> **If PyQtGraph isn't installed** (or is installed but fails to load, for example
> because one of its dependencies is missing), the plot area shows the actual reason,
> the graph controls stay disabled, and the 3-D browser view keeps working regardless.

The map refreshes itself: when you come to the Map section after an analysis, it
notices the new `cluster_info.json` and reloads (Reload is still there for edits
made outside the app). A click *pins* a track in the details panel, so moving the
mouse across other dots on the way to its Play button no longer swaps the track out.
Hiding a cluster in the legend also removes its tracks from the current selection.
The plot follows the app's theme.

The **Open the map** button on Analyse's Results tab switches to Sound Map → Map (an
older version only showed a message box asking you to click the sidebar yourself).
Panels don't switch themselves — the button raises a `navigate_requested` signal and
the main window performs the move, keeping navigation in one place.

### The 3-D view, in your browser

The Map's **Open 3D view** button opens a second, three-dimensional picture of the
same analysis. It is not drawn inside the app: it is a standalone web page,
`Docs/cluster_graph.html`, that `cluster_graph_3d.py` writes from your most recent
analysis (each analysis writes it, and the button regenerates it if it is missing),
and it opens in your default browser as a Three.js/WebGL 3-D scatter plot you can
orbit.

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

## A few historical notes

If you read older material or the git history: there used to be a second, simpler
clustering panel (`clustered.py`) that nothing could reach, and an in-app 3-D plot
widget that was built but never wired in. Both have been deleted. The Analyse section
(`clustered_enhanced.py`) is the only clustering panel, and the browser page above is
the only 3-D view.

*(On the quality report: the dialog that displays your Silhouette /
Davies-Bouldin / Calinski-Harabasz scores used to fail to render its per-cluster
breakdown against any real result — it called an internal method that didn't exist
under that name, and later rendered as three empty boxes because each section
attached a second layout to its card. Both are fixed; the scoring math was never
affected. Likewise, the wizard used to discard every choice (it saved the page being
entered rather than the one being left) and sent "k-means", which ran HDBSCAN; it
now keeps every choice and runs the algorithm you picked.)*

## What's real today, and what's still ahead

In the spirit of an honest, up-to-date resource:

**Working today:** rule-based playlists and Auto-DJ; MusicBrainz-based genre
tag-filling (see the caveat above on what "normalization" means here); clustering on any
of the six sound features (timbre, tempo, harmony, brightness, energy, percussive
density) with both K-Means and HDBSCAN; the five-step configuration wizard;
the quality scores and the report dialog that displays them; **Save cluster
playlists** — a previewed one-playlist-per-cluster export into a per-run folder,
optionally offered right after an analysis; exporting a cluster or any selection
as a playlist from the map; the in-app 2-D map (Sound Map → Map) with
pan/rectangle/lasso selection, hover and click inspection, legend-driven cluster
visibility, and selection → Player / CSV / `.m3u`; and the 3-D view in the browser.

**Still ahead (see ROADMAP.md):** a "smooth path through the cluster" playlist order,
saving a cluster from the legend's right-click, playing a cluster from inside the save
preview, and retiring earlier cluster playlist sets; the wizard's clean-up and dimensionality-reduction options; live re-tuning of a clustering without recomputing
everything from scratch; choosing the map-flattening method from the interface; in-map
cluster *editing* (merging clusters, moving a track between them); selecting by
distance from a point and filtering the map by metadata; a suggestion engine for weak
groupings; and a one-click way to export the quality report.

---

*This concludes the feature walkthroughs. For everything that is planned but not yet
finished across the whole application, see **ROADMAP.md**.*
