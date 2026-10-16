// MathJax 4 configuration for pymdownx.arithmatex (generic mode).
//
// The MathJax version is pinned in mkdocs.yml (extra_javascript) and the font package is pinned here to the
// same release, so a new MathJax release on the CDN cannot change how the book renders without a commit.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  output: {
    font: "mathjax-newcm",
    fontPath: "https://cdn.jsdelivr.net/npm/@mathjax/%%FONT%%-font@4.1.3",
    // Long displayed equations (PTDF, MODF and BSDF updates) break across lines instead of overflowing on
    // narrow screens.
    displayOverflow: "linebreak",
    linebreaks: {
      inline: true,
      width: "100%"
    }
  }
};

// No document$ re-typeset hook: navigation.instant is disabled (print-site does not support it), so every page is
// a full load that MathJax typesets once on startup. Re-typesetting that page from a document$ subscription runs a
// second pass over already-rendered math, which breaks MathJax 4's CHTML layout (fractions and subscripts lose
// their stacking). If navigation.instant is ever enabled, re-add a hook that skips the first emission.
