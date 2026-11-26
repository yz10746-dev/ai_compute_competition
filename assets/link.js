/* Hover highlight only. It does not filter. restyle runs through Plotly,
   so plotly_afterplot must not store the hovered opacity as the base. */
(function () {
  const COMPANIES = new Set([
    "Google", "Meta", "Microsoft", "Amazon", "Oracle",
    "CoreWeave", "xAI", "OpenAI", "Anthropic",
  ]);

  function companyOf(point) {
    const data = point && point.customdata;
    if (typeof data === "string" && COMPANIES.has(data)) return data;
    if (Array.isArray(data)) {
      const name = data[1] || data[0];
      if (COMPANIES.has(name)) return name;
    }
    const named = point && point.data && point.data.name;
    return COMPANIES.has(named) ? named : null;
  }

  function hoveredCompany(event) {
    const points = event.points || [];
    if (!points.length) return null;
    const mouseY = event.event && event.event.clientY;
    let best = points[0];
    let bestDist = Infinity;
    points.forEach((point) => {
      const box = point.bbox;
      if (!box || mouseY == null) return;
      const mid = (box.y0 + box.y1) / 2;
      const dist = Math.abs(mid - mouseY);
      if (dist < bestDist) {
        bestDist = dist;
        best = point;
      }
    });
    return companyOf(best);
  }

  function graphs() {
    return ["area", "share-bar", "score", "raw"]
      .map((id) => document.querySelector("#" + id + " .js-plotly-plot"))
      .filter(Boolean);
  }

  function remember(gd) {
    if (!gd || !gd.data) return;
    gd.__ink = gd.data.map((trace) => ({
      opacity: trace.opacity == null ? 1 : trace.opacity,
      width: trace.marker && trace.marker.line ? trace.marker.line.width : 1,
      color: trace.marker && trace.marker.line ? trace.marker.line.color : "#ffffff",
    }));
  }

  let painting = false;
  let current = null;

  function restyle(gd, update) {
    painting = true;
    const done = () => {
      painting = false;
    };
    const pending = window.Plotly.restyle(gd, update);
    if (pending && pending.then) pending.then(done, done);
    else setTimeout(done, 30);
  }

  function paint(company) {
    if (company === current) return;
    current = company;
    graphs().forEach((gd) => {
      if (!gd.__ink || gd.__ink.length !== gd.data.length) remember(gd);
      const id = gd.parentElement && gd.parentElement.id;
      if (id === "area") {
        restyle(gd, {
          opacity: gd.data.map((trace, index) => {
            if (!company) return gd.__ink[index].opacity;
            return trace.name === company ? 1 : 0.16;
          }),
        });
        return;
      }
      if (id === "share-bar") {
        restyle(gd, {
          "marker.line.width": gd.data.map((trace, index) => {
            const name = trace.customdata && trace.customdata[0];
            if (!company) return gd.__ink[index].width;
            return name === company ? 2.5 : 1;
          }),
          "marker.line.color": gd.data.map((trace, index) => {
            const name = trace.customdata && trace.customdata[0];
            if (!company) return gd.__ink[index].color;
            return name === company ? "#111113" : "#ffffff";
          }),
        });
        return;
      }
      const rows = (gd.data[0] && gd.data[0].customdata) || [];
      restyle(gd, {
        "marker.line.width": [rows.map((row) => (!company || row[1] !== company ? 0 : 1.6))],
        "marker.line.color": ["#111113"],
      });
    });
    document.querySelectorAll(".year-table tbody tr").forEach((row) => {
      const cell = row.querySelector(".dash-cell");
      const name = cell ? cell.innerText.trim() : "";
      row.classList.toggle("row-hot", Boolean(company) && name === company);
    });
  }

  function bind(gd) {
    if (!gd || gd.__linked) return;
    gd.__linked = true;
    remember(gd);
    gd.on("plotly_hover", (event) => paint(hoveredCompany(event)));
    gd.on("plotly_unhover", () => paint(null));
    gd.on("plotly_afterplot", () => {
      if (painting) return;
      remember(gd);
    });
  }

  function watch() {
    graphs().forEach(bind);
  }

  document.addEventListener("mouseover", (event) => {
    const row = event.target.closest && event.target.closest(".year-table tbody tr");
    if (!row) return;
    const cell = row.querySelector(".dash-cell");
    const name = cell ? cell.innerText.trim() : "";
    if (COMPANIES.has(name)) paint(name);
  });
  document.addEventListener("mouseout", (event) => {
    const row = event.target.closest && event.target.closest(".year-table tbody tr");
    if (!row) return;
    const next = event.relatedTarget && event.relatedTarget.closest
      ? event.relatedTarget.closest(".year-table tbody tr")
      : null;
    if (next === row) return;
    paint(null);
  });

  new MutationObserver(watch).observe(document.body, {childList: true, subtree: true});
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watch);
  } else {
    watch();
  }
})();
