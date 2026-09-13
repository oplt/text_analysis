# Frontend UX Redesign Plan — Phase 1 Audit Baseline

**Status:** Phases 1–26 complete  

### Phase 26 implementation notes

Final designer pass (checklist in `fe_prompt.txt` Phase 26). Findings fixed:

- **Grouping / primary actions:** Moved `Next` + `Ask Corpus` into `ResearchContextBar` actions (no orphan toolbar above context).
- **Duplicate chrome:** Removed layout-level Analysis button strip (conflicted with `AnalysisView` `PageTabs` and added scroll).
- **Card weight / scroll:** RAG project scope is an inline filter above tabs (not a full `SectionCard`).
- **Repetition:** Dashboard corpus summary no longer repeats KPI document/unit counts; dropped redundant Coding-progress tip; Classification setup no longer restates corpus/codebook/unit from the context bar; Reliability ready-state dropped duplicated annotator hint.
- **Hierarchy / density:** Research memos accordion title demoted to subtitle/secondary; Analysis group tab label casing aligned (`Statistical models`).
- **Already coherent (verified, no code change):** PageTabs + AdvancedSettings + Provenance drawer patterns; dark theme surfaces (Phase 23); responsive 1024/mobile (Phase 22); run status + disabled reasons (Phases 16/20); Playwright UX suite (Phase 25).

### Phase 25 implementation notes

- New Playwright suite `frontend/e2e/ux-redesign.spec.ts` covering Phase 25 redesign regressions:
  app chrome / skip link, grouped research nav + page tabs, context-change confirm dialog, empty states, help tooltip focus/open, Prepare run-status, annotation / classification / RAG smoke, and mobile (390) / tablet (768) / desktop (1440) responsive shells.
- Optional screenshot baselines behind `E2E_VISUAL=1` (`toHaveScreenshot` with `maxDiffPixelRatio: 0.08`).
- Viewport coverage uses in-spec `setViewportSize` (390 / 768 / 1440); CI stays chromium-only.
- CI `playwright-research` job now also runs `e2e/ux-redesign.spec.ts`.

### Phase 24 implementation notes

- Removed dead Vite template CSS (`App.css`), obsolete `ResearchWorkflowStrip`, and deprecated `AdminSettingsTabs` alias (callers use `SettingsTabs`).
- Slimmed research shared module: deleted unused selectors / re-exports; renamed to `NoCorpusEmptyState.tsx`.
- `RunStatusChip` imports go to `components/ui/RunStatusChip` (no ResearchShared barrel).
- Merged `scientificWarnings.ts` into `ScientificWarnings.tsx` (collector + UI in one module).
- Added `ScrollRegion` primitive; adopted in Topic diagnostics, classification eval, comparative, prediction sets instead of repeated `overflowX: auto` Boxes.

### Phase 23 implementation notes

- Shared `themeSurfaces.ts`: `surfaceCard` / `surfaceMuted` / `surfaceRaised` / `surfaceCode` / `surfaceSelected`, sticky edge shadows, chart series + heat fills — prefer over hard-coded `colors.lightAsh` / hex.
- Theme: mode-aware `action` + `text.disabled`; Dialog / Chip / Alert / Tooltip / TableCell borders & fills; text buttons use `text.secondary` (not fixed pewter).
- Wired surfaces into SectionCard, StatCard, JsonBlock (bordered code), DataTable sticky edges, ResearchCharts (line/scatter/network/heatmap), notifications, dashboard tiles.
- Semantic status colors remain distinct in dark (success/warning/error/info via palette).

### Phase 22 implementation notes

- Documented QA matrix in `responsiveQa.ts`: 390 / 768 / 1024 / 1280 / 1440 / 1920 with phone/tablet/laptop/desktop bands.
- Dense KPI / card / form grids (`ResponsiveCardGrid`, `MetricGrid`, `FormGrid` 3–4 / `4-4-4`, `responsiveColumns`) stay **2 columns through md** and only expand at **lg (1200+)** so 1024–1440 laptop content (~800px beside nav) is not crushed.
- Ask Corpus remains drawer below `lg`; `WorkspaceSplit` `sideFrom="lg"` + `minWidth: 0` / full-width clamps; Ask drawer paper `100%` / `min(100vw, 400px)`.
- Horizontal scroll hardening: `scrollContainerSx`, `DataTable` overflow-x, scrollable `PageTabs`, research nav touch scrolling, chart/matrix wrappers.
- Context bar confirm dialog `fullWidth`; shared selectors use `formControlSx` instead of fixed `minWidth` that overflowed narrow stacks.

### Phase 21 implementation notes

- Accessibility pass on redesign primitives + app chrome (not a full WCAG certification).
- Theme: stronger `:focus-visible` rings on Button / IconButton / Tab / ListItemButton / Chip / Accordion; reduced-motion disables smooth scroll, transitions, and Skeleton animation; caption color no longer hardcodes pewter (inherits theme text); dark-mode input borders/placeholders improved.
- Landmarks: `SkipToContentLink` → `#main-content`; AppLayout `main` is focusable target; `WorkspaceSplit` uses `section` (avoids nested `main`).
- Status not color-alone: `RunStatusChip` text + `aria-label`; `RunStatusPanel` `role="status"` / `aria-live` while active + labelled progress.
- Drawers/tooltips: Provenance drawer labelled dialog + close control + keyboard trigger; HelpTooltip `aria-controls` / dialog paper labelling; AdvancedSettings unique `useId` accordion ids; Tooltips default `describeChild`.
- Errors/loading already announce via QueryBoundary `role="alert"` / `aria-busy` (Phase 20).

### Phase 20 implementation notes

- Shared `DisabledWithReason` wraps disabled primary actions so MUI tooltips still work (span wrapper).
- `actionDisabledReasons.ts` centralizes copy for train / reliability / segment / analysis / robustness / comparative / agent / predict / topics / AL assign / lifecycle / drift.
- `QueryBoundary` now sets `aria-busy` / `role="status"` on loading, `role="alert"` on errors, and a default empty Alert when `isEmpty` has no custom fallback.
- Wired disabled tooltips into Classification, Reliability, Prepare, Analysis, Robustness, Comparative, Agent, Active Learning, Topics, Model Registry.
- Reliability empty/ready copy includes multi-annotator hint (`RELIABILITY_ANNOTATOR_HINT`).

### Phase 19 implementation notes

- Shared `ProvenanceDrawer` + `ProvenancePanel` + `provenanceModel.ts` (friendly labels for run ID, corpus, snapshot, codebook, preprocessing, model, seed, revisions, created by/date).
- Same interaction across research: icon opens drawer; panel used inline in Runs / Methods / Model Registry.
- Fetches `/runs/{id}/provenance` when opened; raw payload under Advanced with Copy JSON.
- Wired into analysis result panel (via MethodsAndProvenanceDrawer), Runs detail, Classification train/predict, Reliability, Robustness, Model Registry.

### Phase 18 implementation notes

- Shared `JsonBlock` (with Copy JSON) + `jsonDisplay` helpers (`recordToKeyValueItems`, `formatDisplayValue`).
- `ResultsInspector` now opens under collapsed **Raw · …** `AdvancedSettings` everywhere (analysis, runs, classification, models, drift, reliability, etc.).
- Primary dumps replaced with KeyValueList / chips / readable text: agent output, dictionary hierarchy/exclusions, export manifest, AI eval cases, playground previews, cleaning custom-regex editor.
- Expert JSON retained under Advanced / Raw / Provenance with copy.

### Phase 17 implementation notes

- Shared `DataTable` + `TruncatedCell` + `dataTableTokens` standardize header style, row density (compact/comfortable), sticky header/first/actions columns, sort, client filter, pagination, selection, column visibility, expansion hook, empty/loading.
- Long cell text truncates with tooltip preview instead of blowing out layout.
- Adopted in corpus documents workspace, analysis `ResearchResultsTable`, Runs list, and RAG documents table.

### Phase 16 implementation notes

- Canonical status model in `runStatusModel.ts`: Queued / Running / Completed / Failed / Cancelled (aliases: pending→queued, processing→running, indexed→completed, etc.).
- Shared `RunStatusChip` + `RunStatusPanel` (status, progress, stage, started, duration, failure, retry/replay, actions slot).
- Wired into: segmentation (`PrepareView`), cleaning apply (`CleaningPanel`), analysis results (`ResearchResults` / `advancedAnalysisShared`), classification train/predict, reliability, robustness, active learning, RAG ingestion (`RagView`), agent Run tab + history (`AgentView`).
- `runPolling.isActiveRunStatus` delegates to the canonical active check.
- Preprocessing remains profile/preview (no async corpus job); uses the same chip when runs appear elsewhere.

**Source roadmap:** `fe_prompt.txt`  
**Audited tree:** `frontend/src/pages/`, `frontend/src/features/`, `frontend/src/features/text-research/`, `frontend/src/components/`  
**Date:** 2026-09-13

### Phase 2 implementation notes

- Grouped research nav: Overview / Data / Coding & Quality / Analysis / Models / Outputs (`researchNavigation.ts`, `ResearchNavBar.tsx`).
- URLs preserved (`/prepare?tab=ingestion|cleaning|preprocessing` for Data sub-destinations).
- Removed floating Stages FAB and analysis “related links” that duplicated destinations.
- Workflow strip no longer primary IA; Progress drawer holds grouped destinations + pipeline stages.
- Context header answers: project, corpus, phase, suggested next step.
- Ask Corpus inline from `lg` (was `xl`).

### Phase 3 implementation notes

- Layout tokens: `components/ui/layoutTokens.ts` (page/section/card spacing + heading hierarchy).
- Extended `PageHeader` (status, primary/secondary actions, overflow menu).
- New primitives: `ResponsiveCardGrid`, `MetricGrid`, `KeyValueList`, `AdvancedSettings`, `ContextInspector`.
- `PageShell` / `SectionCard` / `EmptyState` / `StatCardSkeletonGrid` consume shared tokens.
- Wired: research Ask Corpus → `ContextInspector`; dashboard KPIs → `MetricGrid`; metrics JSON → `KeyValueList`.

### Phase 4 implementation notes

- `ResearchContextBar` shows Project / Corpus / Unit type / Codebook above research navigation.
- Confirm dialogs prevent silent corpus/codebook/unit switches; toast + highlight mark changes.
- Narrow screens collapse to chips with expand-to-edit; Ask Corpus uses `variant="summary"`.
- Removed duplicated project/corpus header + footer context strip from `ResearchLayout`.

### Phase 5 implementation notes

- `components/ui/HelpTooltip.tsx` + `config/helpText.ts` registry (short tooltip / long popover).
- Wired into reliability stats, classification/DFM fields, AL/drift/robustness/runs/model lifecycle, and `KeyValueList` metric keys.

### Phase 6 implementation notes

- Shared `FormGrid` (6/6, 4/4/4, 8/4, …) and `WorkspaceSplit` (main + inspector 8/4).
- KPI rows use `MetricGrid` (1 → 2 → 4); removed hardcoded `repeat(4, 1fr)` on app/project dashboards.
- Research layout uses `WorkspaceSplit`; annotation/corpus/filters/classification/DFM/AL/drift/robustness/context bar rebalanced.

### Phase 7 implementation notes

- URL-backed task tabs via `PageTabs` + `useTabQueryParam` (aliases for legacy `?tab=` values).
- Corpus: Documents / Metadata / Text Units / Quality / Import (`management`→Import, `prepare`→Text Units).
- Prepare: Segment / Ingestion QA / Cleaning / Preprocessing (unchanged).
- Reliability: Overview / Agreement / Disagreements / By Coder / Methods & Provenance (`adjudication`/`history` aliases).
- Analysis: six workspace groups with method sub-tabs; deep links like `?tab=kwic` still resolve.
- Classification: Setup / Dataset / Train / Evaluation / Predictions / Advanced (`evaluate`/`models`/`active` aliases).
- RAG: Documents / Indexing / Retrieval / Evaluation / Runs; Agent: Configure / Run / Trace / Sources / Output.
- Advanced bootstrap params live in `AdvancedSettings` accordions on Reliability.

### Phase 8 implementation notes

- Research `DashboardView` redesigned as a workspace overview (not a metric dump / tabbed JSON panel).
- Responsive 6-KPI `MetricGrid` (3-up desktop): documents, text units, annotation progress, codebooks, trained models, latest run status.
- Sections: Research Progress (workflow stage + blockers), Corpus Summary (units/languages/metadata completeness), Coding Progress (annotation + agreement highlights), Model Summary (version / macro F1 / lifecycle), Recent Activity (runs with workspace links).
- Dashboard API enriched with `language_counts`, `metadata_completeness`, `recent_runs`, and model `lifecycle_status` / `macro_f1`.
- Raw reliability/model JSON replaced with `KeyValueList` readable values.

### Phase 9 implementation notes

- Corpus Documents tab is a Zotero-style master-detail workspace (`CorpusDocumentsWorkspace` + `WorkspaceSplit` 8/4).
- Inline `ContextInspector` shows title, organization, year, language, source, notes, ingestion status, segmentation summary, provenance, and source text.
- Long metadata / provenance live in `AdvancedSettings` accordions; filters use a search+sort toolbar with collapsible facets.
- Sticky table headers, compact/comfortable density, keyboard navigation (↑↓ / Space / Esc / Enter on mobile).
- Mobile keeps the drawer overlay; citation opens from Ask Corpus still use `CorpusDocumentDrawer`.

### Phase 10 implementation notes

- Annotation workspace prioritizes unit text → label decisions → optional confidence/context (accordions).
- Progress bar + status chips (task status, blind / AI-assisted, labels decided); Disagreements deep-link to Reliability.
- Expandable codebook guide sidebar; shortcuts: ←/k · →/j · Tab · y/n/u · ⌘/Ctrl+S.
- Blind vs AI-assisted modes use help terms + distinct alerts; setup mode help wired via `HelpTooltip`.
- Codebook labels use expandable rows (name/definition summary) + inspector editor instead of stacked full cards.

### Phase 11 implementation notes

- Reliability overview/agreement show mean κ / Fleiss / α as readable `MetricCards` with heuristic captions (`reliabilityInterpretation.ts`) plus explicit discipline/use-case caveats (`agreement_benchmarks` help term).
- Coverage cards include labels evaluated, max n_coders, and sum of pairable units; per-label table adds CI, heuristic reading, n_coders.
- Tabs remain Overview / Agreement / Disagreements / By Coder / Methods & Provenance; Methods emphasizes bootstrap CI settings + raw provenance payloads.
- By-coder cards surface primary metric with CI and careful verbal reading; do not treat Landis & Koch–style bands as universal cutoffs.

### Phase 12 implementation notes

- Analysis laboratory layout: `AnalysisLabShell` = configuration (top/main) + results + sticky `Method & provenance` inspector (`WorkspaceSplit` 8/4).
- Workspace groups: Corpus overview / Frequencies / Associations / Statistical Models / Measurement / Advanced (method sub-tabs retained).
- Chart chrome standardized via `AnalysisChartFrame` (title, subtitle, legend, axis cues, export, methodological note) + axis/series labels on `RankedBarChart` / `DivergingBarChart`.
- Method notes live in `analysisMethodMeta.ts`; inline `MethodsAndProvenanceContent` mirrors the drawer for reproducibility without leaving the workspace.

### Phase 13 implementation notes

- Classification sequence: Setup (task type / labels / model family) → Dataset (sample size, visual class distribution, imbalance warnings, planned train/val/test shares) → Train (primary hyperparameters + seed; rare knobs in `AdvancedSettings`) → Evaluation (readable metrics + confusion matrix; diagnostics collapsed) → Predictions → Advanced (raw config/JSON + active learning).
- `ClassificationConfigPanel` no longer expands every hyperparameter by default; feature engineering, selection, calibration, and search live under Advanced Settings.
- Help terms: `class_imbalance`, `class_weight`, `train_test_split`, `validation_set`, `model_family`.
- Evaluation is not JSON-first: train/val/test size cards, metric cards with tooltips, chart frames for F1/confusion; raw payloads moved to Advanced.

### Phase 14 implementation notes

- RAG workspace tabs clarified as Documents → Indexing → Retrieval → Evaluation → Runs with pipeline-oriented copy (`RagView`).
- Documents table shows parsing + indexing states; Indexing surfaces embedding/chunk provenance from chunk metadata, live job progress, and chunk browser.
- Retrieval adds top-k control and a ranked table (source, page, similarity score, snippet).
- Evaluation computes citation coverage / grounding notes from ask responses (`ragEvaluationSummary.ts`); Runs show timestamp, model, project/index scope, status, duration.
- Help terms: `citation_coverage`, `grounding`, `parsing_state`, `indexing_state` (plus existing chunk/embedding/top-k tooltips).

### Phase 15 implementation notes

- Agent workspace is explicitly non-chat: Configure → Run → Trace → Sources → Output.
- Configure surfaces model/providers, tool chips (generate / RAG retrieve / memory), prompt, top-k limits, and retrieval document scope.
- Run shows unified `RunStatusPanel` (task, stage, progress, elapsed, status) plus history table.
- Trace builds structured steps from persisted run fields (`agentTraceModel.ts`) — plan, tool calls, results, errors — without a default raw log dump.
- Sources lists retrieved evidence IDs; Output isolates the final artifact (JSON behind Advanced Settings).

---

## 1. Executive summary

The product already has a solid shared UI kit (`PageShell`, `SectionCard`, `PageTabs`, `EmptyState`, `QueryBoundary`, `StatCard`, `PageHeader`) and consistent MUI usage. The dominant UX debt is concentrated in **Text Research**: flat 19-item tab inventory vs a 15-stage workflow strip, mega-views (1k–1.7k LOC), JSON-first result surfaces, and breakpoint gaps that keep Ask Corpus drawer-only until `xl` (1536px).

Global app chrome (dashboard, projects, calendar, profile, admin) is comparatively thin and coherent. AI surfaces (`/rag`, `/memory`, `/agent`, `/ai`) are reachable by route but mostly absent from primary nav, which hides them from researchers.

**Do not treat this document as permission to restyle.** Later phases in `fe_prompt.txt` own IA (Phase 2), primitives (Phase 3), and page redesigns (Phase 8+).

---

## 2. Global information architecture

### 2.1 Top-level routes (`frontend/src/app/router.tsx`)

| Route | Page / view | Purpose |
|-------|-------------|---------|
| `/` | `AuthHomePage` | Sign-in / marketing shell |
| `/reset-password`, `/verify-email` | Auth pages | Account recovery / verification |
| `/dashboard` | `DashboardPage` | Workspace overview + calendar strip |
| `/calendar` | `CalendarPage` | Full calendar |
| `/projects`, `/projects/:id` | Projects | Project list + detail/tasks |
| `/research` | `ResearchLandingPage` | Pick/resume research project |
| `/research/:projectId/*` | `ResearchPage` → `ResearchLayout` | Full text-research workspace |
| `/platform` | Platform settings | Org/platform configuration |
| `/ai` | `AiStudioView` | Prompt / run playground |
| `/rag` | `RagView` | RAG documents + query |
| `/memory` | `MemoryView` | Memory store CRUD |
| `/agent` | `AgentView` | Agent runs |
| `/observability` | Observability | Health / investigation |
| `/profile`, `/notifications` | Profile / inbox | Account + alerts |
| `/admin/*` | Admin users / platform / settings | Admin-only |

### 2.2 Global navigation (`AppLayout.tsx`)

Primary sidebar (permanent from `md` / 900px; temporary drawer below):

- Dashboard → `/dashboard`
- Projects (label from core-domain config) → `/projects`
- Text Research → last project dashboard or `/research`
- Settings → `/profile`

Toolbar also exposes Calendar, Notifications, theme toggle. **Not in primary nav:** `/ai`, `/rag`, `/memory`, `/agent`, `/observability`, `/platform`, admin routes (admin may appear elsewhere by role).

### 2.3 Research navigation (dual models)

**Flat tabs** — `RESEARCH_TABS` in `types.ts` (19 slugs):  
Dashboard, Corpus, Annotation, Codebook, Reliability, Analysis, Dictionaries, Comparative, Classification, Active Learning, Model Registry, Predictions, Drift, Topics, Robustness, Explorer, Contextual, Runs, Exports.

**Workflow stages** — `RESEARCH_WORKFLOW_STAGES` in `workflow.ts` (15 stages):  
Corpus → Prepare → Codebook → Annotate → Reliability → Analyze → Topics → Classify → Models → Predictions → Drift → Validate → Explore → Contextual → Export.

Mismatches:

| Issue | Detail |
|-------|--------|
| `prepare` | In workflow + router; **not** in `RESEARCH_TABS` |
| Dashboard / Dictionaries / Comparative / Active Learning | In tabs; **not** workflow stages |
| Label drift | Workflow “Validate” → route `robustness`; “Annotate” → `annotation` |
| Extra chrome | `ResearchLayout` analysis sublinks + Dictionaries/Comparative related links |
| Stages FAB | Fixed vertical “Stages” button + `ResearchWorkflowDrawer` duplicates strip |

Chrome stack today: workflow strip + floating Stages + Ask Corpus button/drawer + optional inline Ask Corpus (`xl+`) + memos + citation drawer.

---

## 3. Page inventory

### 3.1 App / platform pages

| Page | LOC (approx) | Major components | Navigation | Length / scroll | Notes |
|------|-------------:|------------------|------------|-----------------|-------|
| `AuthHomePage` | 353 | `AuthShell`, marketing panel | Public | Moderate | Brand-led auth |
| `DashboardPage` | 318 | `PageShell`, calendar, stats | Global | Moderate | Shared primitives |
| `ProjectsPage` | 301 | List + create | Global | Moderate | |
| `ProjectDetailView` | (feature) | Tasks, flow sections | Via projects | Can grow | Cards + drawers |
| `CalendarPage` | 66 | Calendar feature | Global | Moderate | Thin page wrapper |
| `ResearchLandingPage` | 127 | Project picker | Global → research | Short | |
| `ResearchPage` | 1 | Provider + layout | Nested | — | Shell only |
| `ProfileView` | thin page | MFA, account | Settings | Moderate | |
| `NotificationsPage` | 308 | List items | Toolbar | Moderate | |
| `PlatformView` / admin | thin wrappers | Sections | Settings/admin | Dense forms | |
| `ObservabilityView` | feature | Health grid, shortcuts | Hidden from main nav | Moderate | |
| `AiStudioView` | 47 page + panels | Prompt / run / retrieval panels | Hidden | Tabbed | |
| `RagView` | 575 | Tables, forms, status chips | Hidden | Long | Parallel status styling |
| `MemoryView` | 412 | Tables, dialogs | Hidden | Moderate | |
| `AgentView` | 296 | Runs + detail JSON | Hidden | Moderate | `AgentRunDetail` dumps JSON |

### 3.2 Text Research views (`features/text-research/views/`)

| View | LOC | Purpose | Major components | Current nav | UX smells |
|------|----:|---------|------------------|-------------|-----------|
| `ResearchLayout` | 380 | Shell: workflow, Ask Corpus, outlet | Strip, drawers, memos, citation drawer | All research | Dual nav + Stages FAB + `xl` assistant |
| `DashboardView` | 285 | Project research summary | `StatCard`, JSON `<pre>` metrics | Tab/dashboard | **4-col grid no collapse**; raw JSON |
| `CorpusView` | 571 | Corpora + documents | Upload, filters, doc drawer, tabs | Tab + workflow | Prepare overlap with `/prepare` |
| `PrepareView` | 437 | Segmentation / units | Cleaning/prep panels, tabs | Workflow only | Easy to miss from flat tabs |
| `CodebookView` | 565 | Labels / versions | Dialogs, tables | Tab + workflow | |
| `AnnotationView` | 639 | Setup + coding workspace | `PageTabs`, large workspace card | Tab + workflow | Workspace monolith |
| `ReliabilityView` | 1284 | Agreement + adjudication | Tabs, heatmaps, tables, inspectors | Tab + workflow | Excessive vertical scroll |
| `AnalysisView` | 1765 | 15 quantitative methods | `PageTabs`×3 branches, charts, filters | Tab + sublinks | **Critical overload** |
| `DictionaryManagerView` | 567 | Dictionaries | Forms + hierarchy JSON | Tab only | Raw JSON edit/display |
| `ComparativeAnalysisView` | 396 | Prevalence comparison | Forms, results | Tab + related | |
| `ClassificationView` | 1651 | Train / eval / predict | Stepper + tabs + config panel | Tab + workflow | Stacked cards; JSON results |
| `ActiveLearningView` | 502 | AL queue | Cards, tables | Tab only | |
| `ModelRegistryView` | 1113 | Lifecycle + diagnostics | Filters, tables, chips, JSON | Tab + workflow | Scroll farm; chip inconsistency |
| `PredictionSetsView` | 538 | Prediction sets | Filters, tables | Tab + workflow | |
| `DriftMonitoringView` | 625 | Drift metrics | Charts, tables | Tab + workflow | |
| `TopicsView` | 1029 | Topic models | 8 tabs, diagnostics panels | Tab + workflow | Dense diagnostics tables |
| `RobustnessView` | 358 | Robustness checks | Forms, results | Workflow “Validate” | Terminology mismatch |
| `ExplorerView` | 575 | Cross-result explore | Drawers, tables | Tab + workflow | |
| `ContextualView` | 537 | Mixed-method join | Tabs, forms | Tab + workflow | |
| `RunsView` | 642 | Run list / compare / replay | Table, `RunStatusChip`, JSON×N | Tab only | Compare + JSON heavy |
| `ExportsView` | 400 | Exports | Tabs, manifest JSON | Tab + workflow | |

Supporting panels with high reuse / length: `PreprocessingPanel` (766), `TopicDiagnosticsPanels` (848), `ClassificationConfigPanel` (603), `CorpusDocumentDrawer` (363), `ResearchCharts` (387), `ResearchResults` (317), `ResearchWorkflowStrip` (228), `ResearchWorkflowDrawer` (209).

---

## 4. Shared UI pattern audit

### 4.1 Design primitives (`components/ui/`)

| Primitive | Role | Adoption |
|-----------|------|----------|
| `PageShell` | Width / density wrapper | Widespread |
| `PageHeader` | Title + description + actions | Present but **incomplete vs Phase 3 needs** (no status, overflow menu, primary/secondary split) |
| `SectionCard` | Card sections | Very common; overused → “stacked cards” smell |
| `PageTabs` | URL-friendly tabs | Research + some AI |
| `EmptyState` | Empty UX | Good coverage in research; uneven elsewhere |
| `QueryBoundary` | Loading/error boundary | Common |
| `StatCard` / skeleton grid | KPI tiles | Dashboard-oriented |

**Missing / Phase 3 candidates:** `ResponsiveCardGrid`, `MetricGrid`, `KeyValueList`, `AdvancedSettings`, `ContextInspector` (Ask Corpus is bespoke today).

### 4.2 Interaction patterns (research-heavy)

| Pattern | Observation |
|---------|-------------|
| **Cards** | Default container for almost every section; pages become card stacks |
| **Forms** | Large config panels (`ClassificationConfigPanel`, `PreprocessingPanel`, `CleaningPanel`) with fixed `minWidth` fields |
| **Tables** | MUI `Table` everywhere; **no DataGrid**; density/sticky headers inconsistent |
| **Tabs** | Overused as IA within pages (Analysis 15, Topics 8, Classification 5) |
| **Drawers** | Workflow, Ask Corpus, document, explorer — good pattern, inconsistent widths |
| **Dialogs** | Codebook, corpus, dictionaries, memory — OK |
| **Tooltips** | Sparse in research (~16 matches); advanced metrics under-explained |
| **Filters** | Multiple one-offs (`MetadataFilterBar`, corpus filters, registry/runs/predictions inline) |
| **Charts** | Centralized in `ResearchCharts.tsx` — good reuse candidate already |
| **Empty states** | `EmptyState` used; some views still use plain Typography |
| **Loading** | `QueryBoundary` + sparse `Skeleton`; little progress UX outside runs/assistant |
| **Errors** | Many local `<Alert>`; `ScientificWarnings` parallel channel |

### 4.3 Raw JSON in primary UI

| Location | Role |
|----------|------|
| `ResearchShared.JsonBlock` / `ResultsInspector` | Default result dump across analyses/runs |
| `ResearchResults.tsx` | Parameters + metrics JSON |
| `DashboardView.tsx` | Reliability + model metrics `<pre>` |
| `DictionaryManagerView.tsx` | Hierarchy / exclusions |
| `ExportsView.tsx` | Manifest |
| `CleaningPanel.tsx` | Editable custom regex as JSON text |
| `AgentRunDetail.tsx`, `RunPlaygroundPanel.tsx` | Agent/AI output |

This is the primary “scientist console” smell called out for Phase 18.

---

## 5. Repeated patterns → reuse candidates

| Today (duplicated) | Consolidate toward (later phases) |
|--------------------|-----------------------------------|
| `RunStatusChip` vs RAG/model lifecycle `Chip`s | Unified status chip (run / ingestion / lifecycle) |
| `MetadataFilterBar`, corpus filters, inline registry/runs filters | Shared filter row + facet hook |
| Config mega-panels (classification, preprocessing, cleaning, annotation setup) | Shared form section + `AdvancedSettings` accordion |
| `ResultsInspector` / `ResearchResultPanel` / inline `<pre>` | Single results panel: metrics → chart → table → JSON accordion |
| Active run polling + cancel UI scattered | `ActiveRunActions` + shared hook as standard toolbar |
| `CorpusSelector` / `CodebookSelector` / `UnitTypeSelector` | Already shared — enforce adoption |
| Workflow strip + drawer + FAB | One responsive workflow nav spec (Phase 2) |
| Ask Corpus layout | Promote to `ContextInspector` (Phase 3) |

---

## 6. Breakpoint findings

MUI defaults in use: `sm` 600, `md` 900, `lg` 1200, `xl` 1536.

| Target | Code behavior | Problem |
|--------|---------------|---------|
| **390px** | App hamburger; workflow collapses to summary; Stages FAB fixed vertical; Ask Corpus / doc drawers full width | FAB overlaps content; 4-col dashboard grid crushes; horizontal tables overflow |
| **768px** | Still `<md` for app sidebar + workflow strip | Same mobile workflow; drawers ~400px; filter `minWidth: 170` forces wrap/scroll |
| **1024px** | Permanent sidebar; **15-stage horizontal strip** with `overflowX: auto`; Ask Corpus still drawer | Stage labels truncated/scrolled; analysis tab bars wrap heavily |
| **1280px** | `lg` layout; Ask Corpus **still not inline** (`xl` only) | Common laptop widths miss side context; researchers toggle drawer constantly |
| **1440px** | Still `<xl` (1536) | Inline Ask Corpus still off; workflow strip detail gated on `xl` |

Code anchors:

- `ResearchLayout.tsx` — `breakpoints.up("xl")` for inline Ask Corpus; fixed Stages button `writingMode: vertical-rl`
- `ResearchWorkflowStrip.tsx` — compact until `md`, detail until `xl`, `overflowX: auto`
- `DashboardView.tsx` — `gridTemplateColumns: "repeat(4, minmax(0, 1fr))"` with no responsive collapse
- Config/diagnostics panels — fixed widths + `overflowX: auto`

---

## 7. Usability themes (cross-cutting)

1. **Too many equal-weight destinations** — 19 research tabs + 15 stages + analysis sublinks + related links.
2. **Excessive vertical scrolling** — Reliability, Classification, Model Registry, Analysis, Topics.
3. **Stacked cards without hierarchy** — SectionCard wrapping SectionCard; weak primary action.
4. **Poor row balance** — multi-column forms/filters with fixed mins; dashboard 4-up.
5. **Raw JSON** — results and dashboards.
6. **Duplicated explanations** — workflow stage descriptions vs page headers vs helper text.
7. **Unclear terminology** — Validate vs Robustness; Annotate vs Annotation; Prepare missing from tabs; scientific jargon without progressive disclosure.
8. **Hidden AI routes** — RAG/Memory/Agent/AI Studio absent from global nav.

---

## 8. Prioritized findings

### Critical

| ID | Finding | Evidence | Likely phase |
|----|---------|----------|--------------|
| C1 | Dual research navigation (tabs vs workflow) with incomplete overlap | `types.ts`, `workflow.ts`, `ResearchLayout.tsx` | Phase 2 |
| C2 | Analysis workspace cognitive overload (15 method tabs + layout sub-nav + 1765 LOC) | `AnalysisView.tsx` | Phase 2 + 12 |
| C3 | JSON-first results as default research surface | `ResearchCharts` / `ResearchResults` / `RunsView` | Phase 18 done |

### High

| ID | Finding | Evidence | Likely phase |
|----|---------|----------|--------------|
| H1 | Ask Corpus inline only at `xl` (1536px); 1280–1440 stuck on drawer | `ResearchLayout.tsx` | Phase 2–3, 22 |
| H2 | 15-stage horizontal scroller on md–lg; mobile Stages FAB overlaps | `ResearchWorkflowStrip.tsx`, `ResearchLayout.tsx` | Phase 2, 22 |
| H3 | Dashboard KPI grid not responsive | `DashboardView.tsx` | Phase 8, 22 |
| H4 | Mega-view monoliths (Classification, Reliability, Model Registry, Topics) | LOC table §3.2 | Phase 10–13, 24 |
| H5 | Prepare / corpus / cleaning paths duplicated and easy to miss | `PrepareView`, `CorpusView`, workflow | Phase 2, 9 |

### Medium

| ID | Finding | Evidence | Likely phase |
|----|---------|----------|--------------|
| M1 | Inconsistent status visualization across research / RAG / models | `RunStatusChip`, `RagView`, `ModelRegistryView` | Phase 16 |
| M2 | Filter bars reinvented per view | Multiple filter components | Phase 3, 17 |
| M3 | Sparse loading / progress outside QueryBoundary | Few Skeleton usages | Phase 20 |
| M4 | AI/RAG/Memory/Agent not in global nav | `AppLayout.tsx` vs router | Phase 2, 14–15 |
| M5 | Terminology mismatches (Validate/Robustness, etc.) | workflow labels | Phase 2, 5–6 |
| M6 | Dense footer / context metadata in research layout | `ResearchLayout.tsx` | Phase 4–5 |

### Low

| ID | Finding | Evidence | Likely phase |
|----|---------|----------|--------------|
| L1 | No DataGrid — acceptable if intentional; tables lack shared density | Feature-wide | Phase 17 (DataTable) |
| L2 | Inline `JSON.stringify` where `JsonBlock` already exists | Dashboard, Dictionary | Phase 18 |
| L3 | `PageHeader` missing status / overflow / action split | `PageHeader.tsx` | Phase 3 |
| L4 | Tooltips underused for advanced metrics | Low Tooltip counts | Phase 5, 21 |

---

## 9. Recommended next steps (execution order)

Aligned with `fe_prompt.txt`:

1. **Phase 2** — Grouped research nav; collapse dual models; preserve URLs; remove Stages duplication.
2. **Phase 3** — Extend primitives (`PageHeader`, grids, `KeyValueList`, `AdvancedSettings`, `ContextInspector`).
3. **Phases 4–7** — Visual system, density, copy, motion (after IA stabilizes).
4. **Phases 8–13** — Redesign dashboard → corpus → annotation → reliability → analysis → classification using this inventory.
5. **Phases 14–16** — RAG / agents / unified run status (addresses M1, M4).
6. **Phases 17–22** — Tables, JSON removal, provenance, states, a11y, responsive QA at 390/768/1024/1280/1440.

---

## 10. Audit method notes

- Inventory built from `router.tsx`, `RESEARCH_TABS`, `RESEARCH_WORKFLOW_STAGES`, `AppLayout` nav, and `wc -l` on pages/views.
- Pattern counts from ripgrep over `frontend/src/features` (Drawers/Dialogs/Tabs/Tables/EmptyState/Alerts/JSON).
- Responsive claims derived from `useMediaQuery` / `gridTemplateColumns` / fixed `minWidth` / `overflowX` in research layout and dashboards — **not** from live device lab (Phase 22 owns that).
- No large visual refactors were performed in this phase.

---

## 11. Appendix — Component map (shared)

```
frontend/src/components/
  layout/     AppLayout, SettingsTabs, NotificationNavBadge
  ui/         PageShell, PageHeader, PageTabs, SectionCard,
              EmptyState, QueryBoundary, StatCard, StatCardSkeletonGrid
  auth/       AuthShell, AuthMarketingPanel
  guards/     ProtectedRoute
  dashboard/  DashboardCalendar
  notifications/ NotificationListItem
```

Text-research shared:

```
features/text-research/components/
  ResearchShared.tsx          selectors, JsonBlock, RunStatusChip
  ResearchCharts.tsx          charts + ResultsInspector
  ResearchResults.tsx         result panel
  ResearchWorkflowStrip.tsx
  ResearchWorkflowDrawer.tsx
  CorpusDocumentDrawer.tsx
  MetadataFilterBar.tsx
  ActiveRunActions.tsx
  assistant/*                 Ask Corpus
```
