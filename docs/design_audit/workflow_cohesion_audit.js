export const meta = {
  name: 'alphadex-cohesion-audit',
  description: 'Verify the redesigned pages read as one coherent app, visually and in code, and confirm nothing broke',
  phases: [
    { title: 'Inspect', detail: 'per-page visual + code conformance' },
    { title: 'Compare', detail: 'cross-page cohesion on each design axis' },
    { title: 'Verdict', detail: 'synthesize and rank what still diverges' },
  ],
}

const REPO = '/home/user/Music_Indexer'
// args = { shotDir, pages: [{key, file, archetype}] }
const SHOTS = (args && args.shotDir) || `${REPO}/docs/design_audit/shots/after_full`
const PAGES = (args && args.pages) || []
if (!PAGES.length) throw new Error('no pages passed in args.pages')

const PAGE_SCHEMA = {
  type: 'object',
  properties: {
    page: { type: 'string' },
    conforms: { type: 'boolean' },
    regionOrderObserved: { type: 'array', items: { type: 'string' } },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string' },
          axis: { type: 'string' },
          detail: { type: 'string' },
        },
        required: ['severity', 'axis', 'detail'],
      },
    },
    featuresLost: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['page', 'conforms', 'issues', 'notes'],
}

const AXIS_SCHEMA = {
  type: 'object',
  properties: {
    axis: { type: 'string' },
    consistent: { type: 'boolean' },
    rule: { type: 'string' },
    conformingPages: { type: 'array', items: { type: 'string' } },
    deviations: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          page: { type: 'string' },
          whatItDoesInstead: { type: 'string' },
          severity: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['page', 'whatItDoesInstead', 'severity', 'fix'],
      },
    },
  },
  required: ['axis', 'consistent', 'rule', 'deviations'],
}

// ── Per-page inspection: look at the actual rendering, then the code ────────
phase('Inspect')
log(`Inspecting ${PAGES.length} redesigned pages against the spec`)

const perPage = await parallel(PAGES.map(p => () =>
  agent(
    `Verify ONE redesigned page of the AlphaDEX Qt app against the shared design spec.

PAGE: ${p.key}   FILE: ${REPO}/gui/workspaces/${p.file}   ARCHETYPE: ${p.archetype}

YOU ARE READ-ONLY. Do not modify any file.

1. LOOK AT IT. Use the Read tool on these rendered screenshots — you can see images:
     ${SHOTS}/midnight_${p.key}.png
     ${SHOTS}/pearl_${p.key}.png
   (Some pages have per-tab variants suffixed __0, __1 ... — list the directory and read those too.)
   Judge what you actually SEE: region order, alignment, spacing rhythm, visual hierarchy, whether
   anything is clipped, cramped, empty-looking or misaligned, and whether the light and dark
   renderings are equally considered.

2. READ THE SPEC: ${REPO}/docs/design_audit/DESIGN_SPEC.md — specifically the anatomy for the
   "${p.archetype}" archetype and the global rules section.

3. READ THE CODE: ${REPO}/gui/workspaces/${p.file}, and check for prohibited patterns:
     grep -nE '#[0-9a-fA-F]{6}|font-size:[[:space:]]*[0-9]+px' ${REPO}/gui/workspaces/${p.file}
   and look for emoji inside label strings.

4. CHECK NOTHING WAS LOST: run
     cd ${REPO} && git diff 955f9df -- gui/workspaces/${p.file}
   Did any control, handler or feature that existed before disappear? List it in featuresLost.
   This matters MORE than any style finding.

Report what you actually observed. severity: critical / major / minor.
axis should name the design dimension (header, stepper, buttons, cards, spacing, typography,
status, empty-state, icons, colour). An empty issues array is a valid and good result.`,
    { label: `inspect:${p.key}`, phase: 'Inspect', schema: PAGE_SCHEMA }
  )
))

const pages = perPage.filter(Boolean)
log(`Inspected ${pages.length}/${PAGES.length}; comparing across pages`)

// ── Cross-page comparison, one agent per design axis ────────────────────────
phase('Compare')

const AXES = [
  { key: 'header',     q: 'How does each page introduce itself? Hero card vs bare title vs toolbar; is there a subtitle; is the treatment identical across pages of the same archetype AND recognisably related across different archetypes?' },
  { key: 'progress',   q: 'How does each page express stages and progress? Is there exactly ONE idiom app-wide? Watch for leftovers of the three old idioms: a stepper rail, numbered card headings, a wizard counter, and "Phase A/B/C" naming.' },
  { key: 'buttons',    q: 'Button hierarchy: is there exactly one primary per page/tab; is the commit/destructive colour consistent everywhere (the old app contradicted itself — Duplicates red, Tag Fixer green); are secondary/ghost/segment variants used per the spec rather than pages inventing looks?' },
  { key: 'cards',      q: 'Card usage and titling: are cards used consistently; does every card have a title where the spec requires one; is padding and inter-card spacing uniform; are there untitled slabs?' },
  { key: 'typography', q: 'Typography and spacing: are the objectName typography roles used rather than ad-hoc px font-sizes; is the spacing rhythm uniform; does anything look visually larger/smaller than its peers on other pages?' },
  { key: 'status',     q: 'Status, results and empty states: where does status text live; how are results surfaced; do empty states follow one pattern (the old Graph empty state named the missing dependency and the exact install command — that was the best one)?' },
]

const axisResults = await parallel(AXES.map(ax => () =>
  agent(
    `Audit ONE design axis across ALL redesigned pages of the AlphaDEX Qt app, and judge whether the
app now reads as ONE coherent product on this axis.

AXIS: ${ax.key}
QUESTION: ${ax.q}

YOU ARE READ-ONLY. Do not modify any file.

The authoritative rule for this axis is in ${REPO}/docs/design_audit/DESIGN_SPEC.md — read it first
and quote the rule you are auditing against in the 'rule' field.

LOOK AT THE PAGES. Screenshots are in ${SHOTS}/ as midnight_<key>.png and pearl_<key>.png.
List that directory, then Read a representative spread — you can see images. Compare at least these
pages, which span every archetype:
  indexer, duplicates, library_sync, compression, tag_fixer, genres  (pipeline)
  playlists, clustered                                               (tabbed generator)
  graph, similarity                                                  (canvas + inspector)
  player, tools, help                                                (singletons)

Also read the source where the rendering is ambiguous: ${REPO}/gui/workspaces/*.py

Per-page inspection notes from a prior pass (use as leads, verify yourself):
${JSON.stringify(pages.map(p => ({ page: p.page, issues: p.issues }))).slice(0, 7000)}

Report: is this axis consistent across the app? Which pages conform? For each deviation give the
page, what it does instead, severity (critical/major/minor) and a concrete fix.
Be specific and evidence-based — name the page and what you saw. Do not pad the list; if the axis
is genuinely consistent, say so with an empty deviations array.`,
    { label: `axis:${ax.key}`, phase: 'Compare', schema: AXIS_SCHEMA }
  )
))

const axes = axisResults.filter(Boolean)
log(`Compared ${axes.length} axes; writing verdict`)

// ── Final synthesis ────────────────────────────────────────────────────────
phase('Verdict')

const verdict = await agent(
  `Synthesize a final cohesion verdict for the AlphaDEX redesign and WRITE IT TO
${REPO}/docs/design_audit/COHESION_REPORT.md

You MAY create/overwrite that ONE file. Modify nothing else.

PER-PAGE INSPECTION:
${JSON.stringify(pages).slice(0, 20000)}

PER-AXIS CROSS-PAGE AUDIT:
${JSON.stringify(axes).slice(0, 20000)}

Write a report with:
1. VERDICT — does the app now read as one coherent product? One honest paragraph. Do not inflate:
   if pages still diverge, say so plainly and say where.
2. ANYTHING BROKEN OR LOST — every featuresLost entry and every critical finding, page by page.
   This section leads if it is non-empty; a prettier app that lost a feature is a regression.
3. AXIS SCORECARD — a table: axis, consistent yes/no, how many pages deviate.
4. REMAINING DEVIATIONS — ranked by severity, each with page, what it does instead, and the fix.
5. WHAT IMPROVED — concretely, versus the pre-redesign audit in docs/design_audit/index.html.
6. RECOMMENDED NEXT ACTIONS — ordered, specific, and honest about what was not achieved.

Be a critic, not a cheerleader. The value of this report is in what it catches.
Return a 10-line summary; the file is the deliverable.`,
  { label: 'verdict', phase: 'Verdict', effort: 'high' }
)

return {
  pagesInspected: pages.length,
  pagesConforming: pages.filter(p => p.conforms).length,
  featuresLost: pages.flatMap(p => (p.featuresLost || []).map(f => `${p.page}: ${f}`)),
  criticalIssues: pages.flatMap(p => (p.issues || []).filter(i => i.severity === 'critical').map(i => `${p.page}/${i.axis}: ${i.detail}`)),
  axesConsistent: axes.filter(a => a.consistent).map(a => a.axis),
  axesInconsistent: axes.filter(a => !a.consistent).map(a => a.axis),
  verdict,
}
