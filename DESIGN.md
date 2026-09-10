# Text Analysis application design

This document describes the implemented interface. The source of truth for
tokens and component defaults is `frontend/src/app/designTokens.ts` and
`frontend/src/app/theme.ts`; this file records the product-level rules those
files support.

## Product context

Text Analysis is a signed-in research workspace for projects, corpora,
annotation, analysis, RAG documents, agent runs, and platform settings. The UI
prioritizes dense but readable research work over marketing presentation.

## Application shell

- `AppLayout` provides a persistent workspace navigation drawer and top app
  bar. The drawer can collapse on larger screens and becomes temporary,
  menu-controlled navigation on narrow screens.
- Navigation is organized around the current workspace: dashboard, projects,
  text research, calendar, and settings. Research routes preserve the last
  selected project where possible.
- The app bar contains global controls such as notifications, theme selection,
  and the user menu. It is application chrome, not a transparent marketing
  overlay.
- Pages use responsive MUI stacks and grids. Forms, cards, tables, analysis
  controls, status messages, loading skeletons, errors, and empty states are
  first-class interface states.

## Visual system

The current visual system is a restrained, Tesla-inspired neutral palette
adapted to an application UI, not a Tesla product-site implementation.

| Role | Token / value | Use |
| --- | --- | --- |
| Primary action | Electric Blue `#3E6AE1` | Primary buttons, links, and information states |
| Light surfaces | White / Light Ash `#F4F4F4` | Page and alternate surfaces |
| Dark surfaces | Carbon Dark `#171A20` | Dark-mode background |
| Text | Carbon Dark / Graphite `#393C41` / Pewter `#5C5E62` | Primary, secondary, and tertiary text |
| Boundaries | Cloud Gray `#EEEEEE` | Dividers and subtle separation |
| Semantic feedback | MUI success, warning, and error palettes | Status, validation, and failures |

Light and dark modes are both supported. The user can cycle light, dark, and
system preference from the application shell. Component surfaces, text,
dividers, focus treatment, and semantic colors must use the active MUI theme
rather than hard-coded light-mode assumptions.

## Typography and spacing

- Display headings use `Universal Sans Display` with system fallbacks; body,
  controls, and data views use `Universal Sans Text` with system fallbacks.
- Standard body and control text is 14px. Heading scale ranges from 16px to
  40px with medium weight, preserving hierarchy without oversized display
  treatment.
- Use the established MUI spacing scale and `Stack`/responsive `Box` grids.
  Prefer clear grouping, consistent gaps, and readable line lengths over
  full-viewport decorative layouts.
- Buttons have a 40px minimum height and a 4px radius. Cards use a 12px
  radius. Avoid decorative shadows; use spacing, surfaces, and dividers to
  establish structure.

## Interaction and accessibility

- Interactive color, background, border, and shadow changes use the shared
  330ms motion token and easing. Do not add motion that obscures research data
  or changes task flow.
- Keyboard focus is visible through the shared Electric Blue focus ring.
- Icon-only controls need accessible names and tooltips where the action is not
  obvious.
- Responsive layouts must retain usable touch targets, readable data, and
  access to all navigation. Grids collapse to one column where space requires;
  the navigation drawer becomes a temporary drawer on small viewports.
- Async UI must expose loading, recoverable error, empty, authorization, and
  job-state feedback instead of silently leaving stale content on screen.

## Implementation boundaries

- Prefer components and theme values from MUI plus the shared app tokens.
- Do not introduce full-screen vehicle imagery, marketing carousels, vehicle
  navigation, product-order calls to action, or a persistent chatbot bar;
  those belonged to a stale document and are not application features.
- New feature UI belongs under `frontend/src/features/*` and should compose the
  application shell rather than create a competing page chrome.
