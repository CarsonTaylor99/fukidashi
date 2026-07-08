# Dialogue lettering fonts

The reader letters dialogue with the first font in this stack that loads:

1. **Back Issues BB** (Blambot) — not bundled; see below
2. **Komika Text** (Apostrophic Labs) — bundled, freeware (see KomikaText-README.txt)

## Using Back Issues

Blambot's Back Issues is free for indie/non-profit comic use but only
delivered through their checkout, so it can't be bundled here. To use it:

1. Get the free indie license at https://blambot.com/products/back-issues
   (pick "Non-Profit/Indie Comics use", $0 checkout).
2. Drop the font files into this directory named:
   - `BackIssuesBB.ttf` (or `.otf`) — regular
   - `BackIssuesBB-Bold.ttf` (or `.otf`) — bold, optional
3. Hard-refresh the reader (Ctrl+Shift+R). No other config needed —
   the CSS already points at those filenames and falls back to Komika
   Text while they're absent.
