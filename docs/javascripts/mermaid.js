(function () {
  function isDark() {
    return document.body.getAttribute("data-md-color-scheme") === "slate";
  }

  function buildConfig() {
    return {
      startOnLoad: false,
      // Always render in light mode — diagrams use light fills + dark labels.
      // The .mermaid CSS card forces a white background in both colour schemes
      // so everything stays legible without per-diagram dark-mode overrides.
      theme: "base",
      securityLevel: "strict",
      fontFamily:
        "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      fontSize: 13,
      themeVariables: {
        // Base palette — flows, ER, sequence all inherit these.
        background: "#ffffff",
        mainBkg: "#f8f9fb",
        primaryColor: "#ede9fe",
        primaryBorderColor: "#7c3aed",
        primaryTextColor: "#1e1b4b",
        secondaryColor: "#f0f9ff",
        secondaryBorderColor: "#38bdf8",
        secondaryTextColor: "#0c4a6e",
        tertiaryColor: "#fefce8",
        tertiaryBorderColor: "#eab308",
        tertiaryTextColor: "#713f12",
        lineColor: "#94a3b8",
        textColor: "#1f2937",
        titleColor: "#111827",
        edgeLabelBackground: "#ffffff",
        clusterBkg: "#f8f9fb",
        clusterBorder: "#e2e8f0",
        // Sequence diagrams
        actorBkg: "#ede9fe",
        actorBorder: "#7c3aed",
        actorTextColor: "#1e1b4b",
        signalColor: "#6b7280",
        signalTextColor: "#1f2937",
        labelBoxBkgColor: "#f8f9fb",
        labelTextColor: "#1f2937",
        loopTextColor: "#1f2937",
        noteBkgColor: "#fefce8",
        noteBorderColor: "#eab308",
        noteTextColor: "#713f12",
      },
      flowchart: {
        curve: "basis",
        htmlLabels: false,
        useMaxWidth: true,
        padding: 20,
      },
      sequence: { useMaxWidth: true },
      er: { useMaxWidth: true },
    };
  }

  function resetNode(node) {
    if (!node.dataset.mermaidSrc) {
      node.dataset.mermaidSrc = node.textContent;
    }
    node.removeAttribute("data-processed");
    node.textContent = node.dataset.mermaidSrc;
  }

  function render() {
    if (typeof mermaid === "undefined") return;
    var nodes = Array.from(document.querySelectorAll(".mermaid"));
    if (!nodes.length) return;
    mermaid.initialize(buildConfig());
    nodes.forEach(resetNode);
    mermaid.run({ nodes: nodes }).catch(function (err) {
      console.error("Mermaid render error:", err);
    });
  }

  function schedule() {
    window.requestAnimationFrame(render);
  }

  // Re-render whenever the user toggles between light and dark mode.
  new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      if (mutations[i].attributeName === "data-md-color-scheme") {
        schedule();
        return;
      }
    }
  }).observe(document.body, { attributes: true });

  // MkDocs Material instant navigation fires document$ on each page load.
  if (typeof document$ !== "undefined") {
    document$.subscribe(schedule);
  } else {
    document.addEventListener("DOMContentLoaded", schedule);
  }
})();
