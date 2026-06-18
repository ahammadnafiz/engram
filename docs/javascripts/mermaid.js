(function () {
  function resetDiagram(node) {
    if (!node.dataset.mermaidSource) {
      node.dataset.mermaidSource = node.textContent;
    }

    node.removeAttribute("data-processed");
    node.textContent = node.dataset.mermaidSource;
  }

  function renderMermaid() {
    if (typeof mermaid === "undefined") {
      return;
    }

    const diagrams = Array.from(document.querySelectorAll(".mermaid"));
    if (!diagrams.length) {
      return;
    }

    mermaid.initialize({
      startOnLoad: false,
      // Diagrams ship light fills + dark text and sit on a light card in both
      // color schemes (see extra.css), so always render with a light theme.
      theme: "default",
      securityLevel: "strict",
      fontFamily: "Inter, system-ui, sans-serif",
      // Consistent high-contrast palette for every diagram. These merge into
      // each diagram's own frontmatter themeVariables (the diagram's keys win),
      // so text, edges, and labels stay legible on the light card in both
      // light and dark mode without per-diagram tweaks.
      themeVariables: {
        textColor: "#1f2933",
        primaryTextColor: "#1f2933",
        secondaryTextColor: "#1f2933",
        tertiaryTextColor: "#1f2933",
        lineColor: "#52606d",
        // Sequence diagrams
        actorTextColor: "#1f2933",
        signalColor: "#52606d",
        signalTextColor: "#1f2933",
        labelTextColor: "#1f2933",
        loopTextColor: "#1f2933",
        noteTextColor: "#1f2933",
        noteBkgColor: "#fff8e1",
        noteBorderColor: "#f2c94c",
        // Entity-relationship diagrams
        attributeBackgroundColorOdd: "#ffffff",
        attributeBackgroundColorEven: "#f3f5f7",
      },
      flowchart: {
        curve: "basis",
        htmlLabels: true,
        useMaxWidth: true,
      },
      sequence: {
        useMaxWidth: true,
      },
      er: {
        useMaxWidth: true,
      },
    });

    diagrams.forEach(resetDiagram);
    mermaid.run({ nodes: diagrams }).catch((error) => {
      console.error("Failed to render Mermaid diagrams", error);
    });
  }

  function scheduleRender() {
    window.requestAnimationFrame(renderMermaid);
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(scheduleRender);
  } else {
    document.addEventListener("DOMContentLoaded", scheduleRender);
  }
})();
