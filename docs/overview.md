# AlphaDEX — Overview

*Start here. This document explains what AlphaDEX is, who it's for, and the handful
of ideas you need in your head before any of the other documents make sense. It is
written for a curious human, not a programmer — there is no code here, only concepts.*

---

## What AlphaDEX is

AlphaDEX is a **desktop application for taming a large, messy music collection**.
Think of it as a meticulous personal librarian for the thousands of audio files
that have piled up on your hard drive over the years — files with inconsistent
names, duplicate copies in three different folders, missing or wrong information,
and no sensible order.

It is important to understand what AlphaDEX is *not*:

- It is **not** a streaming service. Your music lives on your own machine.
- It is **not** a cloud product. There is no account, no server, no website.
- It is **not** a shared, multi-user system. It is a single-person tool that runs
  on your computer and touches only the folders you point it at.

Because there is no server and no online component, there is also nothing to log
into and nothing that phones home. The only thing resembling a database is a small
local cache (more on that below) that exists purely to make the app faster.

## Who it's for

AlphaDEX is for the person who has **outgrown a simple music player**. If your
collection is small and tidy, you don't need it. If you have a sprawling archive of
ripped CDs, downloads, bandcamp purchases, and DJ sets — and the thought of
manually sorting it makes you tired — that is exactly the situation AlphaDEX was
built for.

## The five things it does

AlphaDEX is organized around five major jobs. Each has its own detailed document in
the `features/` folder; this is the one-sentence version of each:

1. **The Indexer** — scans your library and reorganizes it into a clean, consistent
   folder structure (Artist → Album → Track), fixing messy file names along the way.
2. **The Duplicate Finder** — listens to the actual *sound* of your files to find
   copies of the same song, even when their names differ, and helps you remove the
   lower-quality versions.
3. **Library Sync** — merges a new batch of music (a download, a friend's drive)
   into your existing library, adding what's new and upgrading what's better while
   refusing to create duplicates.
4. **The Tag Fixer** — looks up the correct artist, album, and genre information for
   your tracks from online music databases and proposes corrections.
5. **The Playlist Creator** — builds playlists for you, from simple rules (by tempo,
   by energy, by year) all the way up to letting a machine-learning model group your
   tracks by how they actually *sound*.

## The single most important design principle: look before you leap

If you remember nothing else, remember this. **AlphaDEX never rearranges, replaces,
or deletes your files as a surprise.** Every major operation works in two distinct
phases:

1. **The preview.** AlphaDEX figures out exactly what it *would* do and shows you a
   complete, readable report — usually a web page you can open in your browser. No
   file on disk has been touched at this point.
2. **The execution.** Only after you have looked at the preview and explicitly
   approved it does AlphaDEX actually move, copy, or remove anything.

This "look before you leap" contract runs through the entire application. It exists
because the operations AlphaDEX performs are not easily undone — once a thousand
files have been shuffled into new folders, putting them back by hand would be
miserable. The preview is your safety net and your chance to say "no."

A close companion to this principle is **"quarantine, not delete."** When AlphaDEX
decides a file is a redundant duplicate, its default behavior is not to destroy it
but to move it into a holding area (a folder literally named `Quarantine`). Nothing
is permanently lost unless you go out of your way to ask for it. It is the
difference between a customs holding area and an incinerator.

## A few concepts you'll meet everywhere

These terms appear in nearly every other document. Learn them once here.

### Fingerprint
A **fingerprint** is a short numeric summary of what a piece of audio actually
*sounds like* — the melody, harmony, and rhythm — derived directly from the sound
itself, not from the file's name or tags. The analogy is a **DNA sample**: two files
can have completely different names, different tags, and even different file formats,
yet if they are the same recording, their fingerprints will match. This is the secret
behind both the Duplicate Finder and Library Sync. Computing a fingerprint takes real
effort, which is why AlphaDEX remembers them (see "the cache" below).

### Lossless vs. lossy
Audio files come in two broad families. **Lossless** formats (such as FLAC and WAV)
preserve every detail of the original recording, like a perfect master copy.
**Lossy** formats (such as MP3 and AAC) throw away some detail to make the file
smaller, like a good-but-not-perfect photocopy. When AlphaDEX finds two copies of the
same song, it prefers to keep the lossless one — it is the higher-quality version.
This single preference drives a lot of the app's automatic decisions.

### The reserved folders
AlphaDEX sets aside a handful of specially-named folders inside your library and
treats them as off-limits during scans. Each has a clear job:

| Folder | What it's for |
|---|---|
| `Not Sorted` | A personal "leave this alone" zone. Anything you put here is never scanned or moved. |
| `Playlists` | Where your generated playlist files live. |
| `Manual Review` | A waiting room for tracks that are missing the information AlphaDEX needs to file them correctly. |
| `Docs` | Where AlphaDEX writes its reports, logs, and the small cache. |
| `Trash` | Where stray non-music files (stray images, junk) get swept. |
| `Quarantine` | The holding area for duplicate files that lost out to a better copy. |

When you read elsewhere that "the Indexer skips the reserved folders," this is the
list it means.

### The configuration file
All of your settings — your library location, your matching sensitivity, your API
keys — are kept in a single small text file in your home folder named
`.soundvault_config.json`. The name is a historical artifact: the project used to be
called "SoundVault," and the file was deliberately *not* renamed so that existing
users' settings keep working after they upgrade. Wherever this file is mentioned,
just read it as "the place AlphaDEX keeps your preferences."

## Two front doors: the modern app and the legacy app

One practical thing to know up front: AlphaDEX currently has **two different versions
of its user interface**, and they share the same underlying engine.

- **The modern app** (started via `alpha_dex_gui.py`) is the **current, actively
  developed** version. It has a contemporary look, smooth animations, a sidebar of
  workspaces, and the newest features such as the interactive cluster graph. This is
  the version these documents describe by default.
- **The legacy app** (started via `main_gui.py`) is the **original** version. It
  still works and is kept available as a fallback and a reference, but new
  development happens on the modern app.

The modern app's own startup notes say it plainly: "the original Tkinter app remains
available." When the rest of this documentation says "the app" without qualification,
it means the modern one. The reason both exist is covered in **architecture.md** —
in short, the modern interface was a deliberate rebuild on a more capable foundation,
and the old one was kept around rather than thrown away while the new one matures.

---

*Next: read **architecture.md** for how the pieces fit together behind the scenes,
or jump straight to any feature in the **features/** folder.*
