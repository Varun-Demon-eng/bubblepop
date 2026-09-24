// Background Service Worker for Chrome Extension

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "audit-deepfake-context",
    title: "⚡ Audit for Deepfake / AI Content",
    contexts: ["image", "video", "audio", "link"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "audit-deepfake-context") {
    const targetUrl = info.srcUrl || info.linkUrl;
    if (!targetUrl || !tab) return;

    fetch("http://127.0.0.1:8000/api/v1/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ url: targetUrl, depth: "quick" })
    })
    .then(resp => resp.json())
    .then(data => {
      data.src = targetUrl;
      chrome.tabs.sendMessage(tab.id, {
        action: "HIGHLIGHT_MEDIA_RESULTS",
        results: [data]
      });
    })
    .catch(err => {
      console.error("Context menu audit error:", err);
    });
  }
});
