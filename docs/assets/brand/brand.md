# bolo brand assets

A small, deliberately plain identity for an infrastructure tool.

## The mark

Someone speaking: a head in profile with the mouth open, and the voice leaving it as three
level-meter bars. "bolo" (बोलो) is Hindi for "speak", and the bars are the same ones the console
draws for audio levels.

The mark is drawn on a 32-unit grid as flat white shapes with no outlines, and always sits in a
blue rounded square (corner radius 7 of 32). The neck runs off the bottom edge of the tile. Do not
redraw it at other proportions.

There are two drawings:

- `logo-mark.svg`, for 20 px and up: the head and three bars (short, tall, short).
- `logo-mark-small.svg`, for 16 to 20 px (favicons, the console sidebar): a larger head and two
  bars that sit exactly on the 16 px pixel grid. At these sizes the third bar and the round bar ends
  turn to mush, so they are left out.

How it was chosen: six ideas were drawn and looked at on dark and light at 16, 32, 64 and 256 px.
The profile with bars was the only one that read as "a person talking" at every size. Rejected:
a bust with bars (reads as "contact" or "user"), a round head with a wedge mouth (reads as a game
character), a close-up of an open mouth (unreadable below 64 px), a face made of the bars (reads as
a robot or an assistant), and the earlier headline-with-bars mark with a small mouth under it (the
speaker was lost at small sizes).

## The wordmark

"bolo" and "बोलो" side by side on one baseline, separated by a small dot: the Latin in ink, the
Devanagari in the secondary text colour, so the name is read first and its meaning second. The
Devanagari is set a little smaller (53 against 62 units) so that its headline (shirorekha) sits just
above the Latin x-height rather than towering over it.

- Latin: Inter SemiBold, tracked in by 3.5 %.
- Devanagari: Hind SemiBold (Indian Type Foundry). It was designed as a companion to humanist
  Latin sans-serifs and matches Inter's weight and openness; Mukta SemiBold was the runner-up
  (slightly narrower and lighter next to Inter).

Both are licensed under the SIL Open Font License 1.1, which permits this use. The text was shaped
with HarfBuzz (so the vowel signs attach and the headline is continuous: ब + ो + ल + ो) and
converted to outlines; the SVGs carry no font and render the same everywhere. The console draws the
same outlines (`ui/src/components/logo-devanagari.ts`) instead of shipping the font for four glyphs.

Where there is no room (below about 20 px tile height, or the console's header on a phone), use
the mark with "bolo" alone, or the mark alone.

## Files

| File | Use |
| --- | --- |
| `logo-mark.svg` | The mark in its blue tile, for 20 px and up. Works on any background. |
| `logo-mark-small.svg` | The simplified mark for 16 to 20 px. |
| `logo-mark-mono.svg` | Shapes only, `currentColor`. For single-colour contexts (stamps, embossing, terminal art). |
| `logo-wordmark-dark.svg` | Mark, "bolo" and "बोलो" in light ink, for **dark** backgrounds. Transparent. |
| `logo-wordmark-light.svg` | The same in dark ink, for **light** backgrounds. Transparent. |
| `banner-dark.png`, `banner-light.png` | README hero, 1600 × 480 at 2× (3200 × 960 px), rounded corners on a transparent background. |
| `social-preview.png` | 1280 × 640. The repository's social preview and the console's Open Graph image (`ui/public/og.png` is a copy). |
| `button-live-demo.png` | "Try the live demo" button for the README. It does not show the mark. |
| `favicon/favicon.svg` | Preferred favicon: the small drawing (`ui/public/favicon.svg` is a copy). |
| `favicon/favicon.ico` | 16, 32 and 48 px in one file, for anything that still wants `.ico`. |
| `favicon/favicon-16.png`, `-32.png`, `-48.png` | Raster favicons (small drawing). |
| `favicon/apple-touch-icon.png` | 180 px, full-bleed (`ui/public/apple-touch-icon.png` is a copy). |
| `favicon/icon-192.png`, `icon-512.png` | Full-bleed app icons (`ui/public/` has copies, referenced by `site.webmanifest`). |

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

- Inter for everything Latin (SemiBold for headings, Regular for text).
- Hind SemiBold for "बोलो" in the wordmark only. In the console, text in Indian scripts (transcripts,
  language names) is set in Noto Sans for that script (SIL Open Font License 1.1), loaded on demand.
- JetBrains Mono for numbers, model names and code.
- Sentence case. No all-caps headlines.

## Spacing

- Clear space around the mark or wordmark: at least a quarter of the tile's side on every edge.
- In the wordmark the gap between tile and text is 28% of the tile's side. Keep it.
- Minimum sizes: 16 px for the mark (small drawing below 20 px), 20 px tile height for the wordmark.
  Below that use the mark alone.

## The waveform motif

Bars of one width hanging from a top rule, in accent blue at varying opacity, as in the banners.
Use it sparingly, at the top edge of a composition, never behind text.

## Do

- Use the dark wordmark on dark backgrounds and the light wordmark on light ones.
- Keep the tile blue with the white figure, facing right, speaking towards the name.
- Write the name in lower case: bolo.

## Don't

- Don't recolour the tile, add gradients, shadows or outlines, or rotate the mark.
- Don't stretch the wordmark or change the tile-to-text spacing.
- Don't typeset "bolo" or "बोलो" in another face next to the mark; use the outlined wordmark.
- Don't flip the head, close its mouth, or add features (eyes, hair, a body).
- Don't put the wordmark on photographs or busy backgrounds; put it on a flat surface.
- Don't use the mark to imply endorsement of another product.
