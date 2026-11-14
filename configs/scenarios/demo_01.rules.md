The demo scenario is **curated**: seed **1015** is the first seed in 1000 to 3000 whose 14-day schedule passes every rule below.

| Rule | Why | Passed |
|---|---|---|
| An idler fault with onset on day 2 or later and failure between day 9 and day 12 | A full progression is visible | yes |
| Exactly one drive overheat, failing between day 5 and day 13, at least 24 h from the idler failure | A second signature | yes |
| A heat wave overlapping days 3 to 8 that ends before the idler fault's advanced stage | The model must stay calm while it is hot | yes |
| A heavy-load spell covering at least an hour of a night shift (21:00 to 23:00) while running | A visible, harmless spike | yes |
| A stuck sensor on a healthy zone, starting between day 3 and day 10, with 2 h or more of running time | Shows 'sensor suspect' | yes |
| An outage of at least 2 hours after day 1, on a healthy zone | A silent suite must not look healthy | yes |
