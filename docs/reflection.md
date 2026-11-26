# Reflection

Yuehua Zhuang

## Discussion

### Cleveland & McGill

Cleveland and McGill rank position on a common scale, then length, above area and color for a quantity. Month uses horizontal position. Capacity uses vertical position on the time series and bar length on the share bar. A model is named and ordered on the axis. Its normalized rank, or its training compute, parameters, or training data, uses bar length. Company is categorical, so it uses hue, repeated in every view.

### Stevens’ Law

Stevens’ Law says perceived magnitude follows a power of the true magnitude. Length has an exponent near 1, so a longer bar reads in roughly the right proportion. Area has an exponent near 0.7, so a gap in the stacked area looks smaller than the same gap in the data. The area chart carries the shape of the total over time. A company’s share is read from the bar under it.

### Gestalt

Enclosure puts filters, capacity, and models in three cards. Company and year sit in the left card because they change every view. Measure sits with the capacity views. Model tier and statistic sit with the model views. Each model table sits under its chart. Similarity gives one company one hue across the series, the bar, the model marks, and the table row. The Now totals sit in the header, away from the year band, because they are the current snapshot. The gap between the two model columns keeps the two comparisons apart.

### Data-ink

The share bar’s unit title repeated the Measure control and sat on the tick labels, so it was removed. The hover card still gives the full number and the unit when a precise reading is needed.

### Tradeoffs

The area chart and the share bar show the same capacity. The area is the weaker channel for one share, and it is the view that shows time. Aligning the two tables leaves empty space above the shorter frontier chart. A hatch for months after the snapshot is harder to see than a second color, and a second color would have taken the channel that identifies the company. The headline totals stay on the current snapshot, so they do not answer the selected year.

## Final Channel Justification Table

| Data variable | Data type | Channel | Justification |
| --- | --- | --- | --- |
| Company | Categorical | Color hue | Hue is a nominal channel. One hue identifies the company on the time series, the share bar, the model bars, and the year-table rows. |
| Month | Quantitative | Horizontal position | Position is the strongest channel for a quantity. The axis runs from January 2024, or from the selected year when that year is earlier, through January 2030. |
| Capacity measure | Quantitative | Vertical position; bar length | One measure is shown at a time: H100 equivalent, IT power, facility power, buildings, 8-bit throughput, capital cost, compute cost, construction cost, or operating cost. The time series puts it on vertical position. The share bar puts it on length. |
| Model | Categorical | Vertical position | Each bar on the rank chart and the frontier chart is one model. The axis names it and orders the models by the measure, largest at the top. |
| Normalized rank | Quantitative | Bar length; lightness | Rank is the mean of the within-tier percentiles for training compute, parameters, and training data, using the metrics a model reports. Bar length compares models in the selected year. In the rank table, a deeper step of the company hue is a larger company max or median. |
| Training compute, parameters, or training data | Quantitative | Bar length; lightness | The frontier chart shows one of these for frontier models in the selected year. Length is on a log scale. In the frontier table, lightness encodes the company max or median. |
| Year | Ordinal | Column position; position of a shaded band | Columns run from earlier years to later years, and the selected year column is underlined. The band on the time series is the year used by the share bar and both model charts. |
| Past or future, split at 23 Nov 2026 | Categorical | Hatch | Texture is a nominal channel. Months after the snapshot keep the company hue and add a hatch. A dotted line marks the snapshot. |
| Campus size within one company | Ordinal | Lightness of the company hue | When the share bar splits into campuses, the largest keeps that company’s color. Smaller campuses step lighter, so they stay inside one hue. |
| Chip family | Categorical | Pattern | The chip comparison marks NVIDIA, TPU, Trainium, Other, and Unspecified with distinct textures. It appears only when one to three families are selected, so the texture does not sit on the company hue. |
