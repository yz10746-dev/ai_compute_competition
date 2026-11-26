# Usability test

Yuehua Zhuang

The notes below come from running the three tasks while the dashboard was revised. Classmate quotes and screenshots from those sessions were not recorded.

## Tasks

1. For 2026, report Google’s H100 equivalent as a user, and name its highest normalized-rank model.
2. Restrict the model views to frontier models and compare OpenAI with Anthropic on training compute.
3. Select the largest campus in the share bar and name its owner.

## Observations

Task 1. With no company marked, the charts still drew all nine companies. The empty state looked selected. Google as Owner and Google as User were easy to confuse, because the role control sits away from the number it changes.

Task 2. The year tables used one gray ramp, so a high cell and a company color could not be told apart. The two model tables did not start on the same line, so a year had to be matched by eye. Frontier compute was drawn on an axis that started at zero, which crushed the bars into the right edge.

Task 3. The year band sat between the 2026 and 2027 tick labels, so the highlighted year was ambiguous. The hover card used a transparent background and a two-significant-digit value, and the card was clipped by the chart. A click on a company added or removed it from a list, so the other views did not settle on one company.

A chip filter also raised an exception: the chip summary has no power column, and the table formatter assumed a number.

## Issues

- Empty selection was drawn as every focus company.
- Owner and User were easy to mix up with the number they change.
- One gray ramp hid both magnitude and company.
- The model tables and the year band did not line up with the year they described.
- Frontier bars were crushed against the right edge.
- The hover card was faint, abbreviated, and clipped.
- A company click did not settle the other views on that company.
- A missing chip value crashed the table.

## Design changes

- Select None is the default. No company button is lit, and the charts show every company in the current scope. Focus Group and All Companies are one switch.
- OpenAI, Anthropic, and xAI switch the count to User, because they own no campuses.
- Each table row uses that company’s hue. Lightness encodes the cell value. Companies without their own hue share one ramp. The two tables share a row grid, so their headers line up. Rank and frontier axes start beside the data.
- Year labels sit in the middle of the year band. Dragging the time series, or clicking a year in the left rail, sets that year in the share bar, both model charts, and both tables. A year-column header sorts that table.
- The hover card is white, shows the full number and unit, and fits inside the chart. Hovering a company marks the same company on the other views.
- A click on a chart keeps that company in every view. A second click on the same company returns to Select None. Buttons in the left rail add or remove one company.
- Missing chip values render as a dash.

## Still open

The role control and the Now totals sit far from the charts they describe. A reader can treat the Now totals as the selected year. Select None shows the whole scope, so the charts look filtered only after a company is marked. The frontier chart is shorter than the rank chart, so its title sits above an empty gap in order to keep the tables aligned. When one company is expanded to campuses, the names inside the share bar overlap.

## Screenshots

`docs/figures/` has three annotated screenshots of the current interface. `01-overview.png` is the three-card layout. `02-linked.png` is one company selected across the views. `03-hover.png` is the hover card. Before screenshots were not kept.
