document.addEventListener("DOMContentLoaded", () => {
  const SERVER_URL = "http://127.0.0.1:8000";

  // Header & Close Controls
  const closeBtn = document.getElementById("closeBtn");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => window.close());
  }

  // -------------------------------------------------------------------------
  // 1. TABS & SECTION SWITCHING (IMAGE, VIDEO, AUDIO, TEXT)
  // -------------------------------------------------------------------------
  const tabs = {
    Image: document.getElementById("tabImage"),
    Video: document.getElementById("tabVideo"),
    Audio: document.getElementById("tabAudio"),
    Text:  document.getElementById("tabText")
  };

  const sections = {
    Image: document.getElementById("sectionImage"),
    Video: document.getElementById("sectionVideo"),
    Audio: document.getElementById("sectionAudio"),
    Text:  document.getElementById("sectionText")
  };

  function switchTab(targetModality) {
    Object.keys(tabs).forEach(mod => {
      if (mod === targetModality) {
        tabs[mod].classList.add("active");
        sections[mod].classList.remove("hidden");
      } else {
        tabs[mod].classList.remove("active");
        sections[mod].classList.add("hidden");
      }
    });
  }

  if (tabs.Image) tabs.Image.addEventListener("click", () => switchTab("Image"));
  if (tabs.Video) tabs.Video.addEventListener("click", () => switchTab("Video"));
  if (tabs.Audio) tabs.Audio.addEventListener("click", () => switchTab("Audio"));
  if (tabs.Text)  tabs.Text.addEventListener("click",  () => switchTab("Text"));

  // Helper for setup drag & drop + file input for a section
  function setupFileHandling(dropZoneId, fileInputId) {
    const dropZone = document.getElementById(dropZoneId);
    const fileInput = document.getElementById(fileInputId);
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
      }
    });

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "var(--accent-teal)";
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.style.borderColor = "rgba(0, 230, 165, 0.3)";
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "rgba(0, 230, 165, 0.3)";
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });
  }

  // Bind 4 Drop Zones
  setupFileHandling("dropZoneImage", "fileInputImage");
  setupFileHandling("dropZoneVideo", "fileInputVideo");
  setupFileHandling("dropZoneAudio", "fileInputAudio");
  setupFileHandling("dropZoneText",  "fileInputText");

  // -------------------------------------------------------------------------
  // 2. AUDIT BUTTON HANDLERS FOR EACH MODALITY
  // -------------------------------------------------------------------------

  // Image URL Audit
  const btnAuditImage = document.getElementById("btnAuditImage");
  const urlInputImage = document.getElementById("urlInputImage");
  if (btnAuditImage && urlInputImage) {
    btnAuditImage.addEventListener("click", () => {
      const url = urlInputImage.value.trim();
      if (url) analyzeUrl(url);
    });
  }

  // Video URL Audit
  const btnAuditVideo = document.getElementById("btnAuditVideo");
  const urlInputVideo = document.getElementById("urlInputVideo");
  if (btnAuditVideo && urlInputVideo) {
    btnAuditVideo.addEventListener("click", () => {
      const url = urlInputVideo.value.trim();
      if (url) analyzeUrl(url);
    });
  }

  // Audio URL Audit
  const btnAuditAudio = document.getElementById("btnAuditAudio");
  const urlInputAudio = document.getElementById("urlInputAudio");
  if (btnAuditAudio && urlInputAudio) {
    btnAuditAudio.addEventListener("click", () => {
      const url = urlInputAudio.value.trim();
      if (url) analyzeUrl(url);
    });
  }

  // Text Audit
  const btnAuditText = document.getElementById("btnAuditText");
  const textInputArea = document.getElementById("textInputArea");
  if (btnAuditText && textInputArea) {
    btnAuditText.addEventListener("click", () => {
      const text = textInputArea.value.trim();
      if (text) analyzeTextSnippet(text);
    });
  }

  // -------------------------------------------------------------------------
  // 3. API FETCH FUNCTIONS
  // -------------------------------------------------------------------------
  const resultsDrawer   = document.getElementById("resultsDrawer");
  const resModality     = document.getElementById("resModality");
  const resRiskBadge    = document.getElementById("resRiskBadge");
  const resRiskScore    = document.getElementById("resRiskScore");
  const resProgressFill = document.getElementById("resProgressFill");
  const metricsGrid     = document.getElementById("metricsGrid");

  async function uploadFile(file) {
    showLoading("Auditing file: " + file.name + "...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch(`${SERVER_URL}/api/v1/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      displayResult(data);
    } catch (err) {
      showError("Analysis failed: " + err.message);
    }
  }

  async function analyzeUrl(url) {
    showLoading("Auditing URL target media...");
    const formData = new FormData();
    formData.append("url", url);

    try {
      const resp = await fetch(`${SERVER_URL}/api/v1/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      displayResult(data);
    } catch (err) {
      showError("Analysis failed: " + err.message);
    }
  }

  async function analyzeTextSnippet(text) {
    showLoading("Auditing text stylometrics...");
    const formData = new FormData();
    formData.append("text_content", text);

    try {
      const resp = await fetch(`${SERVER_URL}/api/v1/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      displayResult(data);
    } catch (err) {
      showError("Analysis failed: " + err.message);
    }
  }

  // -------------------------------------------------------------------------
  // 4. DISPLAY RESULTS IN THE FORENSIC REPORT DRAWER
  // -------------------------------------------------------------------------
  function displayResult(res) {
    resultsDrawer.classList.remove("hidden");

    // Modality Badge
    resModality.textContent = res.modality;
    resModality.className = "modality-badge " + (
      res.modality === "AUDIO" ? "badge-audio" :
      res.modality === "IMAGE" ? "badge-image" :
      res.modality === "TEXT"  ? "badge-text"  : "badge-video"
    );

    // Risk Score calculation
    const riskPct = res.risk_score !== undefined ? res.risk_score : (res.prob_ai_generated * 100);
    const riskFormatted = riskPct.toFixed(1) + "%";
    resRiskScore.textContent = riskFormatted;
    resProgressFill.style.width = Math.min(100, Math.max(0, riskPct)) + "%";

    // Risk Level Badge
    if (riskPct < 35.0) {
      resRiskBadge.textContent = "LOW RISK (AUTHENTIC)";
      resRiskBadge.className = "verdict-tag verdict-authentic";
      resProgressFill.className = "progress-fill fill-authentic";
    } else if (riskPct < 65.0) {
      resRiskBadge.textContent = "MODERATE RISK";
      resRiskBadge.className = "verdict-tag verdict-moderate";
      resProgressFill.className = "progress-fill fill-moderate";
    } else {
      resRiskBadge.textContent = "HIGH RISK (DEEPFAKE)";
      resRiskBadge.className = "verdict-tag verdict-fake";
      resProgressFill.className = "progress-fill fill-fake";
    }

    // One-Line Reasoning Text
    const resReasoningBox  = document.getElementById("resReasoningBox");
    const resReasoningText = document.getElementById("resReasoningText");
    if (resReasoningText) {
      resReasoningText.textContent = res.reasoning || "Forensic analysis completed.";
    }

    // Metrics Grid
    metricsGrid.innerHTML = "";
    
    // Add Filename card
    if (res.filename) {
      const fileCard = document.createElement("div");
      fileCard.className = "metric-card";
      fileCard.innerHTML = `<div class="metric-label">TARGET SOURCE</div><div class="metric-value" style="font-size:11px; overflow:hidden; text-overflow:ellipsis;">${res.filename}</div>`;
      metricsGrid.appendChild(fileCard);
    }

    // Add Verdict Card
    const verdictCard = document.createElement("div");
    verdictCard.className = "metric-card";
    verdictCard.innerHTML = `<div class="metric-label">PREDICTED VERDICT</div><div class="metric-value" style="font-size:11px;">${res.verdict}</div>`;
    metricsGrid.appendChild(verdictCard);

    // Add Dynamic Metrics
    if (res.metrics) {
      for (const [key, val] of Object.entries(res.metrics)) {
        const formattedKey = key.replace(/_/g, " ").toUpperCase();
        const card = document.createElement("div");
        card.className = "metric-card";
        card.innerHTML = `
          <div class="metric-label">${formattedKey}</div>
          <div class="metric-value" style="font-size:11px;">${val}</div>
        `;
        metricsGrid.appendChild(card);
      }
    }
  }

  function showLoading(msg) {
    resultsDrawer.classList.remove("hidden");
    resModality.textContent = "SCANNING";
    resModality.className = "modality-badge badge-image";
    resRiskBadge.textContent = "PROCESSING";
    resRiskBadge.className = "verdict-tag";
    resRiskScore.textContent = "...";
    resProgressFill.style.width = "50%";
    metricsGrid.innerHTML = `<div class="metric-card" style="grid-column: span 2;"><div class="metric-label">STATUS</div><div class="metric-value">${msg}</div></div>`;
  }

  function showError(msg) {
    resultsDrawer.classList.remove("hidden");
    resModality.textContent = "ERROR";
    resModality.className = "modality-badge badge-audio";
    resRiskBadge.textContent = "AUDIT FAILED";
    resRiskBadge.className = "verdict-tag verdict-fake";
    resRiskScore.textContent = "0.0%";
    resProgressFill.style.width = "0%";
    metricsGrid.innerHTML = `<div class="metric-card" style="grid-column: span 2;"><div class="metric-label">ERROR DETAILS</div><div class="metric-value" style="font-size:11px; color:#FF3B5C;">${msg}</div></div>`;
  }
});

