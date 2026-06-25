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

### Helpers: genre normalization and year gaps

Two smaller assistants round out this level. **Genre normalization** tidies the messy,
inconsistent genre labels in your files by looking each track up in an online music
encyclopedia and writing back a small set of agreed-upon genres — so your library
stops having forty spellings of "Hip-Hop." A **year-gap assistant** helps build
playlists that tell a chronological story across your collection.

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

### Two graphs: the in-app map and the 3-D browser view

There are actually two flavors of the visualization, and they serve different moods:

- **The in-app 2-D map** is built directly into the application window. It's
  immediate, responsive, and lets you interact with your clusters without ever leaving
  the app — hover, click, select, and spin selections off into playlists.
- **A separate 3-D view** is generated as a standalone web page that opens in your
  browser, where you can orbit around your collection in three dimensions. The extra
  dimension reveals relationships that a flat map can flatten away, at the cost of
  living in a browser tab rather than inside the app.

### Why build the map into the app at all?

It would have been easier to just generate a web page and call it done — and indeed
that's what the 3-D view does. But the in-app 2-D map was deliberately built to live
*inside* the application, so that exploring your clusters and acting on them (making a
playlist from a selection) happens in one continuous flow, without bouncing out to a
browser and back. Keeping the interactive experience in one place was judged worth the
extra effort of building a native, in-app graph rather than offloading everything to an
external web page.

## What's real today, and what's still ahead

In the spirit of an honest, up-to-date resource:

**Working today:** rule-based playlists and Auto-DJ; genre normalization; clustering by
timbre and tempo with both K-Means and HDBSCAN; the configuration wizard; the quality
scores; automatic playlist creation from clusters; the flattening of high-dimensional
data into both a 2-D in-app map and a 3-D browser view; and basic interaction with the
map (hover, click, select).

**Still ahead (see ROADMAP.md):** bringing the additional sound features (harmonic
content, brightness, percussive density) into the actual clustering rather than just
the checkboxes; live re-tuning of a clustering without recomputing everything from
scratch; richer selection and editing tools on the graph (merging groups, moving a
track from one group to another); and a one-click way to export the quality report.

---

*This concludes the feature walkthroughs. For everything that is planned but not yet
finished across the whole application, see **ROADMAP.md**.*
