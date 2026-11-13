# bolo brand assets

A small, deliberately plain identity for an infrastructure tool.

## The mark

A headline with four bars hanging from it. It is a level meter turned upside down, and it borrows
the *shirorekha*, the top line that the letters of Devanagari hang from ("bolo" is Hindi for
"speak"). The first, longest bar doubles as the stem of a "b".

The mark is drawn on a 32-unit grid with one stroke weight (2.8 units, round caps) and always sits
in a rounded square (corner radius 7 of 32). Do not redraw it at other proportions.

## Files

| File | Use |
| --- | --- |
| `logo-mark.svg` | The mark in its blue tile. Works on any background. |
| `logo-mark-mono.svg` | Strokes only, `currentColor`. For single-colour contexts (stamps, embossing, terminal art). |
| `logo-wordmark-dark.svg` | Mark and "bolo" in light ink, for **dark** backgrounds. Transparent. |
| `logo-wordmark-light.svg` | Mark and "bolo" in dark ink, for **light** backgrounds. Transparent. |
| `banner-dark.png`, `banner-light.png` | README hero, 1600 × 480 at 2× (3200 × 960 px), rounded corners on a transparent background. |
| `social-preview.png` | 1280 × 640. The repository's social preview and the console's Open Graph image (`ui/public/og.png` is a copy). |
| `favicon/favicon.svg` | Preferred favicon (`ui/public/favicon.svg` is a copy). |
| `favicon/favicon.ico` | 16, 32 and 48 px in one file, for anything that still wants `.ico`. |
| `favicon/favicon-16.png`, `-32.png`, `-48.png` | Raster favicons. |
| `favicon/apple-touch-icon.png` | 180 px, full-bleed (`ui/public/apple-touch-icon.png` is a copy). |
| `favicon/icon-192.png`, `icon-512.png` | Full-bleed app icons. |

The wordmark is set in Inter SemiBold, tracked in by 3.5%, and converted to outlines, so the SVGs
render the same everywhere without the font. (Inter is licensed under the SIL Open Font License
1.1, which permits this.)

## Colour

| Role | Light surfaces | Dark surfaces |
| --- | --- | --- |
| Brand blue (tile, primary buttons) | `#1f62b3` | `#256abf` |
| Accent blue (lines, charts, motif) | `#2a78d6` | `#3987e5` |
| Ink | `#0b0b0b` | `#f0efec` |
| Secondary text | `#63625d` | `#a3a29a` |
| Page | `#f6f6f4` | `#0d0d0d` |
| Card | `#fcfcfb` | `#1a1a19` |
| Hairline | `#e1e0d9` | `#2c2c2a` |

Blue is the only brand colour. Green, amber and red appear in the console, but only to mean good,
warning and bad; they are not brand colours and should not be used decoratively.

## Type

- Inter for everything (SemiBold for headings, Regular for text).
- JetBrains Mono for numbers, model names and code.
- Sentence case. No all-caps headlines.

## Spacing

- Clear space around the mark or wordmark: at least a quarter of the tile's side on every edge.
- In the wordmark the gap between tile and text is 28% of the tile's side. Keep it.
- Minimum sizes: 16 px for the mark, 20 px tile height for the wordmark. Below 20 px use the mark alone.

## The waveform motif

Bars of one width hanging from a top rule, in accent blue at varying opacity, as in the banners.
Use it sparingly, at the top edge of a composition, never behind text.

## Do

- Use the dark wordmark on dark backgrounds and the light wordmark on light ones.
- Keep the tile blue with white strokes.
- Write the name in lower case: bolo.

## Don't

- Don't recolour the tile, add gradients, shadows or outlines, or rotate the mark.
- Don't stretch the wordmark or change the tile-to-text spacing.
- Don't typeset "bolo" in another face next to the mark; use the outlined wordmark.
- Don't put the wordmark on photographs or busy backgrounds; put it on a flat surface.
- Don't use the mark to imply endorsement of another product.
