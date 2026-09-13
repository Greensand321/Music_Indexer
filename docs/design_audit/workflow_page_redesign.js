export const meta = {
  name: 'alphadex-page-redesign',
  description: 'Redesign AlphaDEX workspace pages in parallel against the shared design spec, then adversarially review and remediate each',
  phases: [
    { title: 'Implement', detail: 'one agent per page, one file each' },
    { title: 'Review', detail: 'fresh adversarial pass against the spec' },
    { title: 'Remediate', detail: 'fix what review confirmed' },
  ],
}

const REPO = '/home/user/Music_Indexer'

// args = { pages: [{key, file, archetype, notes}] }
const PAGES = (args && args.pages) || []
if (!PAGES.length) throw new Error('no pages passed in args.pages')

const RULES = `
=== THE JOB ===
You are redesigning EXACTLY ONE page of the AlphaDEX PySide6 desktop app so that it conforms to the
shared design spec. Other agents are redesigning other pages AT THE SAME TIME. Cohesion depends on
every one of us following the spec literally rather than exercising taste.

=== READ FIRST (in this order) ===
1. ${REPO}/docs/design_audit/DESIGN_SPEC.md   <- THE AUTHORITY. Follow it literally.
2. ${REPO}/docs/design_audit/PAGE_CONTRACTS.md <- what your page must keep exposing.
3. Your page's own source.

=== FILE OWNERSHIP — VIOLATING THIS CORRUPTS OTHER AGENTS' WORK ===
You may modify EXACTLY ONE file: your assigned page under ${REPO}/gui/workspaces/.
You must NOT create, modify or delete ANY other file. Specifically forbidden:
  gui/themes/**        (the theme engine — shared, already complete)
  gui/workspaces/base.py, gui/main_window.py, gui/compat.py
  gui/widgets/**, gui/fonts/**, scripts/**
  any backend module at the repo root (music_indexer_api.py, duplicate_consolidation.py, ...)
If the spec seems to require a change to a shared file, DO NOT make it. Note it in
'sharedChangesNeeded' in your result and work around it for now.
Do NOT run git commit, git add, git checkout, or any git command that mutates state.

=== ENGINEERING CONSTRAINTS (non-negotiable) ===
- This is a LAYOUT AND PRESENTATION redesign. Do NOT change business logic, threading, or what the
  page actually does. Every existing feature must still work. Keep all existing worker-thread
  wiring, signal connections and handler behaviour intact.
- Preserve the page's external contract exactly (PAGE_CONTRACTS.md): public methods, signal names
  and signatures, and any attribute the main window touches. Renaming one silently breaks the app.
- NO hardcoded colours. Ever. Use self.tokens (a WorkspaceBase property returning live ThemeTokens)
  and re-apply in on_theme_changed(tokens). The app has 14 themes, light and dark.
- NO new hardcoded px font-size literals in stylesheets. Use the objectName typography roles.
- NO emoji inside label strings.
- Use R (radii) and S (spacing) from gui.themes.effects instead of inventing numbers.
- The sidebar is a FIXED 360px and the window minimum is 900px wide, so your content area can be as
  narrow as ~540px. Your page MUST NOT force horizontal scrolling.
- Keep the file importable without a running QApplication at module scope.

=== DEFINITION OF DONE ===
Run this and it MUST exit 0 before you finish:
    cd ${REPO} && xvfb-run -a python3 scripts/page_smoke.py <YOUR_KEY>
It builds your page in both a dark and a light theme, cycles every tab, fails on exceptions raised
inside Qt paint overrides (Qt swallows those), and fails if your content minimum width would force
horizontal scrolling. If it does not exit 0, your work is not done — fix it and re-run.
Also run: cd ${REPO} && python3 -c "import ast;ast.parse(open('gui/workspaces/<FILE>').read())"
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    page: { type: 'string' },
    smokePassed: { type: 'boolean' },
    smokeOutput: { type: 'string' },
    regionsBuilt: { type: 'array', items: { type: 'string' } },
    componentsUsed: { type: 'array', items: { type: 'string' } },
    contractPreserved: { type: 'array', items: { type: 'string' } },
    behaviourChanges: { type: 'array', items: { type: 'string' } },
    specDeviations: { type: 'array', items: { type: 'string' } },
    sharedChangesNeeded: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['page', 'smokePassed', 'regionsBuilt', 'componentsUsed', 'summary'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    page: { type: 'string' },
    verdict: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string' },
          category: { type: 'string' },
          detail: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'category', 'detail', 'fix'],
      },
    },
  },
  required: ['page', 'verdict', 'findings'],
}

const REMED_SCHEMA = {
  type: 'object',
  properties: {
    page: { type: 'string' },
    fixed: { type: 'array', items: { type: 'string' } },
    skipped: { type: 'array', items: { type: 'string' } },
    smokePassed: { type: 'boolean' },
    summary: { type: 'string' },
  },
  required: ['page', 'fixed', 'smokePassed', 'summary'],
}

log(`Redesigning ${PAGES.length} page(s): ${PAGES.map(p => p.key).join(', ')}`)

const results = await pipeline(
  PAGES,
  // ── 1. implement ──────────────────────────────────────────────────────────
  (p) => agent(
    `${RULES}

=== YOUR ASSIGNMENT ===
PAGE KEY:   ${p.key}
YOUR FILE:  ${REPO}/gui/workspaces/${p.file}   (the ONLY file you may modify)
ARCHETYPE:  ${p.archetype}
NOTES:      ${p.notes}

Apply the DESIGN_SPEC.md anatomy for the "${p.archetype}" archetype to this page. Follow the spec's
region order, component vocabulary and rules literally — do not substitute your own judgement for a
rule the spec already settled.

Work carefully:
1. Read DESIGN_SPEC.md and PAGE_CONTRACTS.md, then your page in full.
2. Plan the new region structure against the spec's anatomy for your archetype.
3. Rewrite the UI construction. Keep every handler, worker thread and signal connection working.
4. Re-run the smoke command until it exits 0.
5. Re-read your own diff adversarially: any hardcoded colour? any px font-size? any emoji in a
   label? did you touch a file you do not own? did you rename anything in the contract?

Report honestly. If you deviated from the spec, say so in specDeviations and explain why. If
something in the spec was impossible for this page, say so rather than silently ignoring it.
Put the final smoke command output in smokeOutput.`,
    { label: `impl:${p.key}`, phase: 'Implement', schema: IMPL_SCHEMA }
  ),

  // ── 2. adversarial review ─────────────────────────────────────────────────
  (impl, p) => {
    if (!impl) return null
    return agent(
      `You are reviewing ONE page of a parallel redesign for spec compliance and breakage.
You are a FRESH pair of eyes. Be skeptical: the implementer graded their own work.

PAGE: ${p.key}   FILE: ${REPO}/gui/workspaces/${p.file}
ARCHETYPE: ${p.archetype}

THE IMPLEMENTER REPORTED:
${JSON.stringify(impl).slice(0, 6000)}

YOU ARE READ-ONLY. Do not modify any file. Inspect and report only.

Check, in priority order:
1. BREAKAGE — run: cd ${REPO} && git diff -- gui/workspaces/${p.file}
   Did they delete or rename a handler, signal, worker-thread connection or public method?
   Cross-check against ${REPO}/docs/design_audit/PAGE_CONTRACTS.md. Did any FEATURE disappear —
   a control that existed before and now does not? List anything removed.
2. FILE OWNERSHIP — run: cd ${REPO} && git status --short
   Did files outside gui/workspaces/${p.file} change? That is a critical finding.
3. SPEC COMPLIANCE — read ${REPO}/docs/design_audit/DESIGN_SPEC.md and verify the page follows the
   region order, component vocabulary and the explicit rules for its archetype.
4. PROHIBITIONS — grep the file for hardcoded colours (#rrggbb or QColor with literal ints),
   px font-size literals in stylesheets, and emoji inside label strings.
5. ROBUSTNESS — run: cd ${REPO} && xvfb-run -a python3 scripts/page_smoke.py ${p.key}
   Confirm it exits 0 yourself; do not take the implementer's word for it.

severity is one of: critical (breaks the app or loses a feature), major (violates a spec rule),
minor (cosmetic or inconsistent). For each finding give a concrete, actionable fix.
verdict: one short paragraph. An empty findings array is a valid and good result.`,
      { label: `review:${p.key}`, phase: 'Review', schema: REVIEW_SCHEMA }
    )
  },

  // ── 3. remediate ──────────────────────────────────────────────────────────
  (review, p) => {
    if (!review) return null
    const actionable = (review.findings || []).filter(f => f.severity !== 'minor')
    if (!actionable.length) {
      return { page: p.key, fixed: [], skipped: [], smokePassed: true, summary: 'review clean — no remediation needed' }
    }
    return agent(
      `${RULES}

=== REMEDIATION ===
PAGE: ${p.key}
YOUR FILE: ${REPO}/gui/workspaces/${p.file}   (still the ONLY file you may modify)

An independent reviewer found these issues in the redesign that was just applied. Fix them.

REVIEWER VERDICT: ${review.verdict}

FINDINGS TO FIX:
${JSON.stringify(actionable, null, 1).slice(0, 8000)}

Fix every critical and major finding. Prioritise anything that lost a feature or broke a contract —
restoring behaviour matters more than spec purity. If a finding is wrong, say so in 'skipped' with
your reasoning rather than making a change you believe is incorrect.

Finish only when this exits 0:
    cd ${REPO} && xvfb-run -a python3 scripts/page_smoke.py ${p.key}`,
      { label: `fix:${p.key}`, phase: 'Remediate', schema: REMED_SCHEMA }
    )
  }
)

const out = []
PAGES.forEach((p, i) => {
  out.push({ page: p.key, remediation: results[i] || null })
})
return { pages: out }
