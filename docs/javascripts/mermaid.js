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
