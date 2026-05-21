---
name: Clinical Precision AI
colors:
  surface: '#f7f9fb'
  surface-dim: '#d8dadc'
  surface-bright: '#f7f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f6'
  surface-container: '#eceef0'
  surface-container-high: '#e6e8ea'
  surface-container-highest: '#e0e3e5'
  on-surface: '#191c1e'
  on-surface-variant: '#434655'
  inverse-surface: '#2d3133'
  inverse-on-surface: '#eff1f3'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#565e74'
  on-secondary: '#ffffff'
  secondary-container: '#dae2fd'
  on-secondary-container: '#5c647a'
  tertiary: '#943700'
  on-tertiary: '#ffffff'
  tertiary-container: '#bc4800'
  on-tertiary-container: '#ffede6'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#dae2fd'
  secondary-fixed-dim: '#bec6e0'
  on-secondary-fixed: '#131b2e'
  on-secondary-fixed-variant: '#3f465c'
  tertiary-fixed: '#ffdbcd'
  tertiary-fixed-dim: '#ffb596'
  on-tertiary-fixed: '#360f00'
  on-tertiary-fixed-variant: '#7d2d00'
  background: '#f7f9fb'
  on-background: '#191c1e'
  surface-variant: '#e0e3e5'
typography:
  display-lg:
    fontFamily: Manrope
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.03em
  headline-lg:
    fontFamily: Manrope
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Manrope
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.02em
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: -0.01em
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: -0.01em
  label-bold:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: '1.4'
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.4'
    letterSpacing: 0.02em
  headline-lg-mobile:
    fontFamily: Manrope
    fontSize: 28px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  gutter: 24px
  margin: 32px
  container-max: 1440px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 32px
---

## Brand & Style
The design system is engineered to evoke absolute trust, analytical rigour, and high-end medical sophistication. It targets healthcare professionals and clinical researchers who require high-density information environments that remain legible and calm under pressure.

The aesthetic follows a **Corporate Modern** foundation infused with **Glassmorphism**. This combination balances the structural reliability of medical instrumentation with a modern, breathable interface. The UI should feel like a premium digital surgical suite: sterile but not cold, complex but hyper-organized, and technologically advanced yet human-centric.

## Colors
The palette is rooted in a "Clinical White" spectrum, using subtle off-whites to distinguish between different functional layers without relying on heavy borders. 

- **Primary (Medical Blue):** Reserved strictly for primary actions, progress indicators, and active states.
- **Deep Charcoal:** Used for primary typography to ensure maximum contrast and an authoritative feel.
- **Semantic Alerts:** High-saturation tones (Amber, Red, Emerald) are used sparingly for clinical data status to ensure they "pop" against the neutral background.
- **Surfaces:** Depth is achieved through a hierarchy of whites rather than grays, maintaining a clean, high-end feel.

## Typography
Typography in this design system prioritizes data density and hierarchy. **Manrope** provides a refined, modern look for headlines, while **Inter** ensures utilitarian precision for clinical data and UI labels.

To achieve the "sophisticated" feel, tracking (letter spacing) is slightly tightened on headlines to create a more compact, intentional visual footprint. We utilize a wide range of weights—from 400 for long-form reading to 700 for critical headers—to ensure that even in data-heavy screens, the most important information is immediately discoverable.

## Layout & Spacing
The layout follows a **Fixed-Fluid Hybrid** model. Dashboards and data grids use a fluid 12-column grid to maximize screen real estate on large medical monitors, while content-heavy pages (like patient reports) conform to a fixed-width central column for readability.

A strict 4px baseline grid governs all spacing, ensuring that all components align with mathematical precision. Margins and gutters are generous (24px+) to prevent the high-density information from feeling overwhelming, creating "visual breathing room" between complex data modules.

## Elevation & Depth
Depth is created through **Glassmorphism and Tonal Layering**. 

1.  **Base Layer:** Solid `#f8fafc` (Neutral).
2.  **Card Layer:** Solid white with a 1px border of `#e2e8f0`.
3.  **Overlay/Modal Layer:** Translucent white (`rgba(255, 255, 255, 0.7)`) with a 20px-32px background blur (backdrop-filter).
4.  **Shadows:** Shadows are highly diffused and tinted with the secondary charcoal color at very low opacity (3-5%) to avoid a "dirty" look, appearing instead as soft ambient occlusion.

This hierarchy ensures that high-priority diagnostic tools or alerts appear to float precisely above the baseline data.

## Shapes
The design system uses **Soft (Level 1)** roundedness. 

A corner radius of 4px-8px is applied to all interactive elements and data containers. This "instrument-grade" rounding is enough to feel modern and approachable without losing the serious, precise character required for medical software. Buttons and inputs use a consistent 4px radius, while larger containers and cards use 8px to softly frame their content.

## Components
- **Buttons:** Primary buttons use the Medical Blue with white text. Secondary buttons use a subtle "Ghost" style—clear backgrounds with a 1px soft border that fills on hover.
- **Clinical Chips:** Small, semi-translucent status indicators. For example, a "Critical" chip uses a 10% opacity Red background with 100% opacity Red text for high legibility without visual "noise."
- **Data Cards:** Utilize a subtle 1px border and a very soft drop shadow. Cards containing high-priority AI insights should utilize the background-blur effect to distinguish them from standard patient data.
- **Input Fields:** Use a minimalist approach—bottom borders only or very light 1px outlines that thicken and change to Medical Blue on focus.
- **Glass Overlays:** Used for sidebar navigation and secondary tool panels to maintain context of the underlying clinical data while the user interacts with system controls.
- **System Status Bar:** A persistent, thin element at the top or bottom using Emerald for "System Ready" states to provide constant reassurance of operational integrity.