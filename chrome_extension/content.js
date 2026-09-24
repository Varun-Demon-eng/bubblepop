// Content Script for Multi-Modal Deepfake Extension

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "COLLECT_PAGE_MEDIA") {
    const mediaItems = [];

    // Collect <img> elements
    document.querySelectorAll("img").forEach((img) => {
      if (img.src && img.width > 50 && img.height > 50) {
        mediaItems.push({ src: img.src, type: "image" });
      }
    });

    // Collect <video> elements
    document.querySelectorAll("video").forEach((vid) => {
      if (vid.src) {
        mediaItems.push({ src: vid.src, type: "video" });
      } else {
        vid.querySelectorAll("source").forEach((src) => {
          if (src.src) mediaItems.push({ src: src.src, type: "video" });
        });
      }
    });

    // Collect <audio> elements
    document.querySelectorAll("audio").forEach((aud) => {
      if (aud.src) {
        mediaItems.push({ src: aud.src, type: "audio" });
      } else {
        aud.querySelectorAll("source").forEach((src) => {
          if (src.src) mediaItems.push({ src: src.src, type: "audio" });
        });
      }
    });

    sendResponse({ mediaItems: mediaItems });
  }

  if (request.action === "HIGHLIGHT_MEDIA_RESULTS") {
    const results = request.results || [];
    results.forEach((res) => {
      if (!res.src) return;

      // Find matching DOM elements
      const elements = document.querySelectorAll(`[src="${res.src}"]`);
      elements.forEach((el) => {
        highlightElement(el, res);
      });
    });
  }
});

function highlightElement(el, res) {
  const isAuth = res.verdict.toUpperCase().includes("AUTHENTIC");
  
  // Create relative wrapper if needed
  if (getComputedStyle(el).position === "static") {
    el.style.position = "relative";
  }

  el.style.outline = isAuth ? "3px solid #00E6A5" : "3px solid #FF3B5C";
  el.style.outlineOffset = "-3px";

  // Create or update badge overlay
  let badge = el.parentElement.querySelector(`.deepfake-badge[data-src="${res.src}"]`);
  if (!badge) {
    badge = document.createElement("div");
    badge.className = "deepfake-badge";
    badge.setAttribute("data-src", res.src);
    badge.style.position = "absolute";
    badge.style.top = "8px";
    badge.style.left = "8px";
    badge.style.zIndex = "9999";
    badge.style.padding = "4px 8px";
    badge.style.borderRadius = "6px";
    badge.style.fontSize = "11px";
    badge.style.fontWeight = "bold";
    badge.style.backdropFilter = "blur(8px)";
    badge.style.boxShadow = "0 4px 12px rgba(0,0,0,0.4)";
    badge.style.color = "#FFFFFF";
    
    if (el.parentElement) {
      el.parentElement.style.position = "relative";
      el.parentElement.appendChild(badge);
    }
  }

  const conf = (res.confidence * 100).toFixed(1);
  if (isAuth) {
    badge.style.background = "rgba(0, 230, 165, 0.9)";
    badge.textContent = `✓ AUTHENTIC (${conf}%)`;
  } else {
    badge.style.background = "rgba(255, 59, 92, 0.9)";
    badge.textContent = `⚠ DEEPFAKE (${conf}%)`;
  }
}
