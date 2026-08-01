/* =====================================================================
   NewsClassify — Main JavaScript
   Handles: footer year, character counter, example news loader,
   clear button, and the AJAX call to /api/predict with result rendering.
   ===================================================================== */

document.addEventListener("DOMContentLoaded", function () {

    // ---------------- Footer year ----------------
    const yearEl = document.getElementById("year");
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }

    // ---------------- Elements (Predict page only) ----------------
    const articleInput = document.getElementById("articleInput");
    const charCount = document.getElementById("charCount");
    const predictBtn = document.getElementById("predictBtn");
    const clearBtn = document.getElementById("clearBtn");
    const errorAlert = document.getElementById("errorAlert");
    const errorMsg = document.getElementById("errorMsg");

    const idleState = document.getElementById("idleState");
    const loadingState = document.getElementById("loadingState");
    const resultState = document.getElementById("resultState");

    const resultCategory = document.getElementById("resultCategory");
    const categoryIcon = document.getElementById("categoryIcon");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");
    const probBreakdown = document.getElementById("probBreakdown");

    // Exit early if we're not on the Predict page
    if (!articleInput) return;

    const MAX_CHARS = 3000;

    // ---------------- Character counter ----------------
    function updateCharCount() {
        const len = articleInput.value.length;
        charCount.textContent = len;
        charCount.style.color = len > MAX_CHARS * 0.9 ? "#ff6b6b" : "";
    }
    articleInput.addEventListener("input", updateCharCount);
    updateCharCount();

    // ---------------- Example news dropdown ----------------
    document.querySelectorAll(".dropdown-item[data-example]").forEach(function (item) {
        item.addEventListener("click", function (e) {
            e.preventDefault();
            articleInput.value = this.getAttribute("data-example");
            updateCharCount();
            hideError();
            articleInput.focus();
        });
    });

    // ---------------- Clear button ----------------
    clearBtn.addEventListener("click", function () {
        articleInput.value = "";
        updateCharCount();
        hideError();
        showIdleState();
        articleInput.focus();
    });

    // ---------------- Error helpers ----------------
    function showError(message) {
        errorMsg.textContent = message;
        errorAlert.classList.remove("d-none");
    }
    function hideError() {
        errorAlert.classList.add("d-none");
    }

    // ---------------- Result panel state helpers ----------------
    function showIdleState() {
        idleState.classList.remove("d-none");
        loadingState.classList.add("d-none");
        resultState.classList.add("d-none");
    }
    function showLoadingState() {
        idleState.classList.add("d-none");
        loadingState.classList.remove("d-none");
        resultState.classList.add("d-none");
    }
    function showResultState() {
        idleState.classList.add("d-none");
        loadingState.classList.add("d-none");
        resultState.classList.remove("d-none");
    }

    // ---------------- Predict button ----------------
    predictBtn.addEventListener("click", function () {
        const text = articleInput.value.trim();
        hideError();

        if (!text) {
            showError("Please enter or paste a news article before predicting.");
            return;
        }
        if (text.length < 20) {
            showError("Please enter a longer article (at least 20 characters) for an accurate prediction.");
            return;
        }

        showLoadingState();
        predictBtn.disabled = true;

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        })
        .then(function (response) {
            return response.json().then(function (data) {
                return { ok: response.ok, data: data };
            });
        })
        .then(function (result) {
            predictBtn.disabled = false;

            if (!result.ok || !result.data.success) {
                showIdleState();
                showError(result.data.error || "Something went wrong. Please try again.");
                return;
            }
            renderResult(result.data);
        })
        .catch(function (err) {
            predictBtn.disabled = false;
            showIdleState();
            showError("Could not reach the server. Please check your connection and try again.");
            console.error(err);
        });
    });

    // ---------------- Render prediction result ----------------
    function renderResult(data) {
        const meta = (typeof CATEGORY_META !== "undefined" && CATEGORY_META[data.category])
            ? CATEGORY_META[data.category]
            : { icon: "fa-tag", color: "#5b3df0" };

        resultCategory.textContent = data.category;
        categoryIcon.innerHTML = '<i class="fa-solid ' + meta.icon + '"></i>';
        categoryIcon.style.background = meta.color;

        confidenceValue.textContent = data.confidence.toFixed(1) + "%";
        confidenceBar.style.width = "0%";
        confidenceBar.style.background = meta.color;

        // Probability breakdown bars
        probBreakdown.innerHTML = "";
        const sorted = Object.entries(data.probabilities).sort(function (a, b) { return b[1] - a[1]; });

        sorted.forEach(function ([label, value]) {
            const rowMeta = (typeof CATEGORY_META !== "undefined" && CATEGORY_META[label])
                ? CATEGORY_META[label]
                : { color: "#5b3df0" };

            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML =
                '<span class="prob-row-label">' + label + '</span>' +
                '<span class="prob-row-track"><span class="prob-row-fill" style="width:0%; background:' + rowMeta.color + ';"></span></span>' +
                '<span class="prob-row-value">' + value.toFixed(1) + '%</span>';
            probBreakdown.appendChild(row);
        });

        showResultState();

        // Animate bars after they're in the DOM
        requestAnimationFrame(function () {
            confidenceBar.style.width = data.confidence + "%";
            document.querySelectorAll(".prob-row-fill").forEach(function (fill, i) {
                fill.style.width = sorted[i][1] + "%";
            });
        });
    }
});
