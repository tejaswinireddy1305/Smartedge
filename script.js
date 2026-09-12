// ============================================================
// SMARTEDGE FRONTEND - FINAL WORKING VERSION
// ============================================================

"use strict";

const API_URL = "http://127.0.0.1:8001";

let benchmarkChart = null;
let hitRateChart = null;
let refreshTimer = null;


// ============================================================
// BASIC HELPERS
// ============================================================

function $(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = $(id);

    if (element) {
        element.textContent = value;
    }
}

function toNumber(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function toPercent(value) {
    const n = toNumber(value);

    return Math.abs(n) <= 1
        ? n * 100
        : n;
}

function formatNumber(value) {
    return toNumber(value).toLocaleString();
}

function formatPercent(value) {
    return `${toPercent(value).toFixed(2)}%`;
}

function escapeHTML(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// ============================================================
// API REQUEST
// ============================================================

async function apiRequest(endpoint, options = {}) {

    let response;

    try {

        response = await fetch(
            `${API_URL}${endpoint}`,
            {
                ...options,

                headers: {
                    "Accept": "application/json",

                    ...(options.body
                        ? {
                            "Content-Type":
                                "application/json"
                        }
                        : {}),

                    ...(options.headers || {})
                }
            }
        );

    } catch (error) {

        console.error(
            "Network error:",
            error
        );

        throw new Error(
            "Cannot connect to SmartEdge backend. " +
            "Make sure FastAPI is running on port 8001."
        );
    }


    let data = null;

    try {

        data = await response.json();

    } catch (error) {

        data = null;
    }


    if (!response.ok) {

        let message =
            `API error: ${response.status}`;


        if (data && data.detail) {

            if (Array.isArray(data.detail)) {

                message =
                    data.detail
                        .map(
                            item =>
                                item.msg ||
                                String(item)
                        )
                        .join(", ");

            } else {

                message =
                    String(data.detail);
            }
        }

        else if (data && data.message) {

            message =
                String(data.message);
        }


        throw new Error(message);
    }


    return data;
}


// ============================================================
// API STATUS
// ============================================================

async function checkAPIConnection() {

    const status =
        $("apiStatus");


    try {

        const data =
            await apiRequest("/health");


        const healthy =
            !data ||
            data.status === "healthy";


        if (status) {

            status.textContent =
                healthy
                    ? "Backend Online"
                    : "Backend Degraded";


            status.classList.toggle(
                "connected",
                healthy
            );


            status.classList.toggle(
                "disconnected",
                !healthy
            );
        }


        return healthy;


    } catch (error) {

        console.error(
            "API connection error:",
            error
        );


        if (status) {

            status.textContent =
                "Backend Offline";


            status.classList.remove(
                "connected"
            );


            status.classList.add(
                "disconnected"
            );
        }


        return false;
    }
}


// ============================================================
// NAVIGATION
// ============================================================

const VIEW_IDS = {

    dashboard:
        "view-dashboard",

    cache:
        "view-cache",

    analytics:
        "view-analytics",

    about:
        "view-about"
};


function normalizePage(page) {

    page =
        String(page || "")
            .replace(/^#/, "")
            .toLowerCase()
            .trim();


    const aliases = {

        home:
            "dashboard",

        main:
            "dashboard",

        analysis:
            "analytics",

        "cache-section":
            "cache",

        "ask-ai":
            "ask-ai",

        chatbot:
            "ask-ai",

        chat:
            "ask-ai",

        prediction:
            "dashboard",

        "ml-prediction":
            "dashboard"
    };


    return (
        aliases[page] ||
        page ||
        "dashboard"
    );
}


function navigate(page) {

    const target =
        normalizePage(page);


    // Ask AI is a section inside Dashboard

    if (target === "ask-ai") {

        showView("dashboard");


        setTimeout(
            () => {

                const aiSection =
                    $("ask-ai");


                if (aiSection) {

                    aiSection.scrollIntoView({
                        behavior:
                            "smooth",

                        block:
                            "start"
                    });
                }

            },
            100
        );


        return;
    }


    showView(target);
}


function showView(page) {

    const target =
        normalizePage(page);


    const validView =
        VIEW_IDS[target]
            ? target
            : "dashboard";


    Object.entries(VIEW_IDS)
        .forEach(
            ([name, id]) => {

                const view =
                    $(id);


                if (!view) {
                    return;
                }


                const active =
                    name === validView;


                view.classList.toggle(
                    "active",
                    active
                );


                view.hidden =
                    !active;


                view.style.display =
                    active
                        ? ""
                        : "none";
            }
        );


    // Active navigation button

    document
        .querySelectorAll(
            ".header-nav .nav-link"
        )
        .forEach(
            link => {

                let linkPage =
                    link.dataset.page ||
                    link.dataset.section ||
                    link.getAttribute(
                        "href"
                    ) ||
                    "";


                linkPage =
                    normalizePage(
                        linkPage
                    );


                const active =
                    linkPage ===
                    validView;


                link.classList.toggle(
                    "active",
                    active
                );


                if (active) {

                    link.setAttribute(
                        "aria-current",
                        "page"
                    );

                } else {

                    link.removeAttribute(
                        "aria-current"
                    );
                }
            }
        );


    // URL hash

    const hash =
        validView === "dashboard"
            ? "#dashboard"
            : `#${validView}`;


    if (
        window.location.hash !==
        hash
    ) {

        history.replaceState(
            null,
            "",
            hash
        );
    }


    // Load page data

    if (
        validView ===
        "dashboard"
    ) {

        loadLiveStats();
        loadCacheContents();
    }


    if (
        validView ===
        "cache"
    ) {

        loadLiveStats();
        loadCacheContents();
    }


    if (
        validView ===
        "analytics"
    ) {

        loadLiveStats();
        loadBenchmark();
    }
}


function initializeNavigation() {

    const links =
        document.querySelectorAll(
            ".header-nav .nav-link"
        );


    links.forEach(
        link => {

            link.addEventListener(
                "click",
                event => {

                    event.preventDefault();


                    const page =
                        link.dataset.page ||
                        link.dataset.section ||
                        link.getAttribute(
                            "href"
                        ) ||
                        "dashboard";


                    navigate(page);
                }
            );
        }
    );


    window.addEventListener(
        "hashchange",
        () => {

            const page =
                window.location.hash ||
                "#dashboard";


            const normalized =
                normalizePage(page);


            if (
                normalized ===
                "ask-ai"
            ) {

                navigate(
                    "ask-ai"
                );

            } else {

                showView(
                    normalized
                );
            }
        }
    );


    const initial =
        window.location.hash ||
        "#dashboard";


    if (
        normalizePage(initial) ===
        "ask-ai"
    ) {

        navigate(
            "ask-ai"
        );

    } else {

        showView(initial);
    }
}


// ============================================================
// LIVE STATISTICS
// ============================================================

async function loadLiveStats() {

    try {

        const data =
            await apiRequest(
                "/cache/stats"
            );


        updateDashboardStats(
            data
        );


        updateHitRateChart(
            data
        );


        setText(
            "ctxHitRate",
            formatPercent(
                data?.hit_rate
            )
        );


        setText(
            "ctxTotalRequests",
            formatNumber(
                data?.total_requests
            )
        );


        setText(
            "ctxCacheSize",
            formatNumber(
                data?.cache_size
            )
        );


        return data;


    } catch (error) {

        console.error(
            "Statistics loading error:",
            error
        );


        return null;
    }
}


// ============================================================
// UPDATE DASHBOARD
// ============================================================

function updateDashboardStats(data) {

    if (!data) {
        return;
    }


    const totalRequests =
        toNumber(
            data.total_requests
        );


    const hits =
        toNumber(
            data.hits
        );


    const misses =
        toNumber(
            data.misses
        );


    const hitRate =
        toPercent(
            data.hit_rate
        );


    const missRate =
        toPercent(
            data.miss_rate
        );


    const cacheSize =
        toNumber(
            data.cache_size
        );


    const capacity =
        toNumber(
            data.cache_capacity,
            100
        );


    setText(
        "totalRequests",
        formatNumber(
            totalRequests
        )
    );


    setText(
        "cacheHits",
        formatNumber(
            hits
        )
    );


    setText(
        "hitRateTrend",
        `${hitRate.toFixed(2)}% hit rate`
    );


    setText(
        "cacheMisses",
        formatNumber(
            misses
        )
    );


    setText(
        "missRateTrend",
        `${missRate.toFixed(2)}% miss rate`
    );


    setText(
        "cacheSize",
        `${cacheSize} / ${capacity}`
    );


    setText(
        "cacheCapacity",
        `Capacity: ${capacity} items`
    );
}


// ============================================================
// HIT / MISS CHART
// ============================================================

function updateHitRateChart(data) {

    const canvas =
        $("hitRateChart");


    if (
        !canvas ||
        typeof Chart ===
        "undefined"
    ) {

        return;
    }


    if (hitRateChart) {

        try {

            hitRateChart.destroy();

        } catch (error) {}

        hitRateChart =
            null;
    }


    hitRateChart =
        new Chart(
            canvas,
            {

                type:
                    "doughnut",

                data: {

                    labels: [
                        "Cache Hits",
                        "Cache Misses"
                    ],

                    datasets: [

                        {
                            data: [

                                toNumber(
                                    data?.hits
                                ),

                                toNumber(
                                    data?.misses
                                )
                            ],

                            borderWidth:
                                1
                        }
                    ]
                },

                options: {

                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    plugins: {

                        legend: {

                            display:
                                true,

                            position:
                                "bottom"
                        }
                    }
                }
            }
        );
}


// ============================================================
// BENCHMARK
// ============================================================

async function loadBenchmark() {

    try {

        const data =
            await apiRequest(
                "/summary"
            );


        updateBenchmark(
            data
        );


        return data;


    } catch (error) {

        console.error(
            "Benchmark loading error:",
            error
        );


        return null;
    }
}


function updateBenchmark(data) {

    if (!data) {
        return;
    }


    const algorithms =
        data.algorithms ||
        {};


    const lru =
        algorithms.LRU ||
        {};


    const lfu =
        algorithms.LFU ||
        {};


    const smartedge =
        algorithms.SmartEdge ||
        {};


    setText(
        "lruRate",
        formatPercent(
            lru.hit_rate
        )
    );


    setText(
        "lruHits",
        formatNumber(
            lru.hits
        )
    );


    setText(
        "lruEvictions",
        formatNumber(
            lru.evictions
        )
    );


    setText(
        "lfuRate",
        formatPercent(
            lfu.hit_rate
        )
    );


    setText(
        "lfuHits",
        formatNumber(
            lfu.hits
        )
    );


    setText(
        "lfuEvictions",
        formatNumber(
            lfu.evictions
        )
    );


    setText(
        "smartedgeRate",
        formatPercent(
            smartedge.hit_rate
        )
    );


    setText(
        "smartedgeHits",
        formatNumber(
            smartedge.hits
        )
    );


    setText(
        "smartedgeInsertions",
        formatNumber(
            smartedge.proactive_insertions
        )
    );


    createBenchmarkChart(
        lru,
        lfu,
        smartedge
    );
}


// ============================================================
// BENCHMARK CHART
// ============================================================

// ============================================================
// BENCHMARK CHART - COLORED VERSION
// ============================================================

function createBenchmarkChart(lru, lfu, smartedge) {

    const canvas = $("benchmarkChart");

    if (
        !canvas ||
        typeof Chart === "undefined"
    ) {
        return;
    }

    // Destroy previous chart
    if (benchmarkChart) {
        try {
            benchmarkChart.destroy();
        } catch (error) {}

        benchmarkChart = null;
    }

    const lruRate =
        toPercent(lru?.hit_rate);

    const lfuRate =
        toPercent(lfu?.hit_rate);

    const smartedgeRate =
        toPercent(smartedge?.hit_rate);


    benchmarkChart = new Chart(
        canvas,
        {
            type: "bar",

            data: {
                labels: [
                    "LRU",
                    "LFU",
                    "SmartEdge"
                ],

                datasets: [
                    {
                        label:
                            "Cache Hit Rate (%)",

                        data: [
                            lruRate,
                            lfuRate,
                            smartedgeRate
                        ],

                        // ====================================================
                        // COLORS - LIKE YOUR 2ND SCREENSHOT
                        // ====================================================

                        backgroundColor: [
                            "#94A3B8",   // LRU - Slate Gray
                            "#536B8A",   // LFU - Dark Blue Gray
                            "#2563EB"    // SmartEdge - Strong Blue
                        ],

                        borderColor: [
                            "#94A3B8",
                            "#536B8A",
                            "#2563EB"
                        ],

                        borderWidth: 1,

                        borderRadius: 0,

                        barPercentage: 0.70,

                        categoryPercentage: 0.75
                    }
                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                animation: {
                    duration: 500
                },

                plugins: {

                    legend: {
                        display: true,

                        position: "top",

                        labels: {
                            usePointStyle: false,

                            boxWidth: 50,

                            padding: 15,

                            font: {
                                size: 14
                            }
                        }
                    },

                    tooltip: {

                        enabled: true,

                        backgroundColor:
                            "rgba(30, 30, 30, 0.95)",

                        titleFont: {
                            size: 14,
                            weight: "bold"
                        },

                        bodyFont: {
                            size: 13
                        },

                        padding: 10,

                        callbacks: {

                            label: function(context) {

                                return (
                                    " Hit Rate (%): " +
                                    Number(
                                        context.raw
                                    ).toFixed(3)
                                );
                            }
                        }
                    }
                },

                scales: {

                    x: {

                        grid: {
                            display: true,

                            color:
                                "rgba(0, 0, 0, 0.08)"
                        },

                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    },

                    y: {

                        beginAtZero: true,

                        max: 100,

                        ticks: {

                            stepSize: 10,

                            callback:
                                function(value) {
                                    return value + "%";
                                },

                            font: {
                                size: 13
                            }
                        },

                        title: {

                            display: true,

                            text:
                                "Hit Rate (%)",

                            font: {
                                size: 14
                            }
                        },

                        grid: {

                            display: true,

                            color:
                                "rgba(0, 0, 0, 0.08)"
                        }
                    }
                }
            }
        }
    );
}


// ============================================================
// CACHE CONTENTS
// ============================================================

async function loadCacheContents() {

    try {

        const data =
            await apiRequest(
                "/cache"
            );


        renderCacheTable(
            data
        );


        return data;


    } catch (error) {

        console.error(
            "Cache loading error:",
            error
        );


        return null;
    }
}


function getCacheItems(data) {

    const raw =
        data?.items ??
        data?.cache ??
        [];


    if (
        Array.isArray(raw)
    ) {

        return raw.map(
            (
                item,
                index
            ) => {

                return {

                    url:
                        item?.url ||
                        item?.key ||
                        item?.content_id ||
                        `Object ${index + 1}`,

                    size:
                        item?.content_size ??
                        item?.content_size_bytes ??
                        item?.size ??
                        0,

                    cachedAt:
                        item?.cached_at ??
                        item?.timestamp ??
                        item?.created_at ??
                        "-",

                    action:
                        item?.action ||
                        "CACHED"
                };
            }
        );
    }


    if (
        raw &&
        typeof raw ===
        "object"
    ) {

        return Object.entries(
            raw
        ).map(
            (
                [url, value]
            ) => {

                let size =
                    0;

                let cachedAt =
                    "-";


                if (
                    value &&
                    typeof value ===
                    "object"
                ) {

                    size =
                        value.content_size ??
                        value.content_size_bytes ??
                        value.size ??
                        0;


                    cachedAt =
                        value.cached_at ??
                        value.timestamp ??
                        value.created_at ??
                        "-";

                } else {

                    size =
                        toNumber(
                            value
                        );
                }


                return {

                    url:
                        url,

                    size:
                        size,

                    cachedAt:
                        cachedAt,

                    action:
                        "CACHED"
                };
            }
        );
    }


    return [];
}


// ============================================================
// RENDER CACHE TABLE
// ============================================================

function renderCacheTable(data) {

    const items =
        getCacheItems(
            data
        );


    const dashboardBody =
        $("cacheTableBody");


    const cachePageBody =
        $("cacheTableBodyAlt");


    const bodies =
        [
            dashboardBody,
            cachePageBody
        ].filter(
            Boolean
        );


    bodies.forEach(
        tbody => {

            if (
                items.length ===
                0
            ) {

                tbody.innerHTML = `

                    <tr>

                        <td
                            colspan="5"
                            class="empty-row"
                        >
                            Cache is currently empty
                        </td>

                    </tr>
                `;

                return;
            }


            tbody.innerHTML =
                items.map(
                    (
                        item,
                        index
                    ) => {

                        const sizeKB =
                            toNumber(
                                item.size
                            ) / 1024;


                        return `

                            <tr>

                                <td>
                                    ${index + 1}
                                </td>

                                <td>
                                    ${escapeHTML(
                                        item.url
                                    )}
                                </td>

                                <td>
                                    ${sizeKB.toFixed(2)}
                                    KB
                                </td>

                                <td>
                                    ${escapeHTML(
                                        item.cachedAt
                                    )}
                                </td>

                                <td>
                                    <strong>
                                        ${escapeHTML(
                                            item.action
                                        )}
                                    </strong>
                                </td>

                            </tr>
                        `;
                    }
                ).join("");
        }
    );
}


// ============================================================
// ML PREDICTION
// ============================================================

async function runPrediction() {

    const hour =
        $("hour");

    const day =
        $("dayOfWeek");

    const frequency =
        $("pastFrequency");

    const recency =
        $("recency");

    const contentSize =
        $("contentSize");


    if (
        !hour ||
        !day ||
        !frequency ||
        !recency ||
        !contentSize
    ) {

        console.error(
            "ML input fields are missing."
        );

        return;
    }


    const request = {

        hour:
            Number(
                hour.value
            ),

        day_of_week:
            Number(
                day.value
            ),

        past_frequency:
            Number(
                frequency.value
            ),

        recency_seconds:
            Number(
                recency.value
            ),

        content_size:
            Number(
                contentSize.value
            )
    };


    if (
        !Number.isFinite(
            request.hour
        ) ||
        request.hour < 0 ||
        request.hour > 23
    ) {

        alert(
            "Hour must be between 0 and 23."
        );

        return;
    }


    if (
        !Number.isFinite(
            request.day_of_week
        ) ||
        request.day_of_week < 0 ||
        request.day_of_week > 6
    ) {

        alert(
            "Day of week must be between 0 and 6."
        );

        return;
    }


    if (
        !Number.isFinite(
            request.past_frequency
        ) ||
        request.past_frequency < 0
    ) {

        alert(
            "Past frequency must be 0 or greater."
        );

        return;
    }


    if (
        !Number.isFinite(
            request.recency_seconds
        ) ||
        request.recency_seconds < 0
    ) {

        alert(
            "Recency must be 0 or greater."
        );

        return;
    }


    if (
        !Number.isFinite(
            request.content_size
        ) ||
        request.content_size <= 0
    ) {

        alert(
            "Content size must be greater than 0."
        );

        return;
    }


    const button =
        $("runPredictionBtn");


    if (button) {
        button.disabled = true;
    }


    try {

        const data =
            await apiRequest(
                "/predict",
                {

                    method:
                        "POST",

                    body:
                        JSON.stringify(
                            request
                        )
                }
            );


        console.log(
            "Prediction:",
            data
        );


        const probability =
            toPercent(
                data?.reusable_probability_percent ??
                data?.reusable_probability ??
                0
            );


        const threshold =
            toPercent(
                data?.threshold_percent ??
                data?.threshold ??
                0.5
            );


        const shouldCache =
            probability >= threshold;


        setText(
            "predictionPercentage",
            `${probability.toFixed(2)}%`
        );


        setText(
            "predictionLabel",

            shouldCache
                ? "REUSE"
                : "NO_REUSE"
        );


        const box =
            $("predictionDecisionBox");


        if (box) {

            box.classList.toggle(
                "cache",
                shouldCache
            );


            box.classList.toggle(
                "rejected",
                !shouldCache
            );
        }


        setText(
            "cacheDecision",

            shouldCache

                ? `CACHE • ${probability.toFixed(2)}% ≥ ${threshold.toFixed(0)}%`

                : `DO NOT CACHE • ${probability.toFixed(2)}% < ${threshold.toFixed(0)}%`
        );


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        setText(
            "predictionPercentage",
            "--"
        );


        setText(
            "predictionLabel",
            "ERROR"
        );


        setText(
            "cacheDecision",
            error.message ||
            "Prediction failed."
        );


    } finally {

        if (button) {
            button.disabled = false;
        }
    }
}


// ============================================================
// LIVE CACHE REQUEST
// ============================================================

async function sendCacheRequest() {

    const urlInput =
        $("requestUrl");

    const sizeInput =
        $("requestContentSize");


    if (
        !urlInput ||
        !sizeInput
    ) {

        console.error(
            "Request input fields missing."
        );

        return;
    }


    const url =
        urlInput.value.trim();


    const contentSize =
        Number(
            sizeInput.value
        );


    if (!url) {

        alert(
            "Please enter a resource URL."
        );

        return;
    }


    if (
        !Number.isFinite(
            contentSize
        ) ||
        contentSize <= 0
    ) {

        alert(
            "Content size must be greater than 0."
        );

        return;
    }


    setText(
        "requestStatus",
        "..."
    );


    setText(
        "requestMessage",
        "Processing request..."
    );


    setText(
        "requestDetails",
        "Checking cache and running ML prediction..."
    );


    try {

        const data =
            await apiRequest(
                "/request",
                {

                    method:
                        "POST",

                    body:
                        JSON.stringify({

                            url:
                                url,

                            content_size:
                                contentSize
                        })
                }
            );


        console.log(
            "Cache request:",
            data
        );


        const cacheStatus =
            String(
                data?.cache_status ||
                "UNKNOWN"
            ).toUpperCase();


        setText(
            "requestStatus",
            cacheStatus
        );


        if (
            cacheStatus ===
            "HIT"
        ) {

            setText(
                "requestMessage",
                "CACHE HIT"
            );


            setText(
                "requestDetails",
                "Resource served directly from SmartEdge edge cache."
            );

        } else {

            setText(
                "requestMessage",
                "CACHE MISS"
            );


            setText(
                "requestDetails",

                data?.message ||
                "Resource was not found in cache."
            );
        }


        const prediction =
            data?.ml_prediction ||
            {};


        const decision =
            data?.cache_decision ||
            {};


        const probability =
            toPercent(
                prediction.reusable_probability_percent ??
                prediction.reusable_probability ??
                0
            );


        const threshold =
            toPercent(
                decision.threshold_percent ??
                decision.threshold ??
                0.5
            );


        const admitted =
            probability >= threshold;


        setText(
            "requestPrediction",

            admitted
                ? "REUSE"
                : "NO_REUSE"
        );


        setText(
            "requestProbability",

            `${probability.toFixed(2)}%`
        );


        setText(
            "requestDecision",

            admitted
                ? "ADMITTED"
                : "REJECTED"
        );


        setText(
            "requestResponseTime",

            `${toNumber(
                data?.response_time_ms
            ).toFixed(2)} ms`
        );


        const details =
            $("requestDetails");


        if (details) {

            details.textContent +=

                ` ML reuse probability: ` +

                `${probability.toFixed(2)}%.` +

                ` Threshold: ` +

                `${threshold.toFixed(0)}%.` +

                (
                    admitted

                        ? " Object admitted to cache."

                        : " Object rejected and NOT stored in cache."
                );
        }


        await Promise.all([
            loadLiveStats(),
            loadCacheContents()
        ]);


    } catch (error) {

        console.error(
            "Cache request error:",
            error
        );


        setText(
            "requestStatus",
            "ERROR"
        );


        setText(
            "requestMessage",
            "Request Failed"
        );


        setText(
            "requestDetails",
            error.message ||
            "Unable to contact SmartEdge API."
        );
    }
}


// ============================================================
// RESET CACHE
// ============================================================

async function resetCache() {

    try {

        await apiRequest(
            "/cache/reset",
            {
                method:
                    "POST"
            }
        );


        setText(
            "predictionPercentage",
            "--"
        );


        setText(
            "predictionLabel",
            "Waiting..."
        );


        setText(
            "cacheDecision",
            "Enter features and run prediction."
        );


        const box =
            $("predictionDecisionBox");


        if (box) {

            box.classList.remove(
                "rejected"
            );


            box.classList.add(
                "cache"
            );
        }


        await Promise.all([
            loadLiveStats(),
            loadCacheContents()
        ]);


    } catch (error) {

        console.error(
            "Reset cache error:",
            error
        );


        alert(
            error.message ||
            "Unable to reset cache."
        );
    }
}


// ============================================================
// ASK SMARTEDGE AI
// ============================================================

async function askSmartEdge() {

    const questionElement =
        $("aiQuestion");


    const answerElement =
        $("aiAnswer");


    const evidenceElement =
        $("aiEvidence");


    const errorElement =
        $("aiError");


    const loadingElement =
        $("aiLoading");


    const agentTag =
        $("aiAgentTag");


    const agentElement =
        $("aiAgent");


    const question =
        questionElement
            ? questionElement.value.trim()
            : "";


    if (!question) {

        if (errorElement) {

            errorElement.textContent =
                "Please enter a question.";

            errorElement.style.display =
                "block";
        }

        return;
    }


    if (errorElement) {

        errorElement.textContent =
            "";

        errorElement.style.display =
            "none";
    }


    if (loadingElement) {

        loadingElement.style.display =
            "block";
    }


    const button =
        $("askAiBtn");


    if (button) {

        button.disabled =
            true;

        button.dataset.oldText =
            button.innerText;

        button.innerText =
            "Analyzing...";
    }


    try {

        console.log(
            "Sending question to SmartEdge AI:",
            question
        );


        // IMPORTANT:
        // Use the same API helper as the rest of the frontend.

        const data =
            await apiRequest(
                "/agent/ask",
                {

                    method:
                        "POST",

                    body:
                        JSON.stringify({
                            question:
                                question
                        })
                }
            );


        console.log(
            "SmartEdge AI response:",
            data
        );


        if (
            data &&
            data.status &&
            data.status !==
            "success"
        ) {

            throw new Error(

                data.message ||

                "SmartEdge AI could not answer the question."
            );
        }


        const result =
            data?.result ||
            {};


        // ----------------------------------------------------
        // ANSWER
        // ----------------------------------------------------

        const answer =

            result.answer ||

            result.recommendation ||

            data?.answer ||

            "No answer was returned by SmartEdge AI.";


        if (answerElement) {

            answerElement.textContent =
                answer;
        }


        // ----------------------------------------------------
        // AGENT
        // ----------------------------------------------------

        const agent =

            data?.selected_agent ||

            result.agent ||

            "SmartEdge AI";


        if (agentElement) {

            agentElement.textContent =
                agent;
        }


        if (agentTag) {

            agentTag.style.display =
                "inline-block";
        }


        // ----------------------------------------------------
        // EVIDENCE
        // ----------------------------------------------------

        if (evidenceElement) {

            let evidence =

                result.evidence ||

                data?.evidence ||

                [];


            if (
                !Array.isArray(evidence)
            ) {

                evidence =
                    [];
            }


            if (
                evidence.length >
                0
            ) {

                evidenceElement.innerHTML =

                    evidence
                        .map(
                            item => {

                                if (
                                    typeof item ===
                                    "string"
                                ) {

                                    return `
                                        <p>
                                            ${escapeHTML(
                                                item
                                            )}
                                        </p>
                                    `;
                                }


                                const source =

                                    item?.source ||

                                    item?.filename ||

                                    item?.file ||

                                    item?.title ||

                                    "Evidence";


                                return `
                                    <p>
                                        <strong>
                                            ${escapeHTML(
                                                source
                                            )}
                                        </strong>
                                    </p>
                                `;
                            }
                        )
                        .join("");

            } else {

                evidenceElement.innerHTML = `

                    <p class="muted">
                        Answer generated from
                        SmartEdge evidence.
                    </p>

                `;
            }
        }


        // ----------------------------------------------------
        // LIVE SYSTEM CONTEXT
        // ----------------------------------------------------

        const liveContext =
            data?.live_context ||
            {};


        const stats =
            liveContext.cache_stats ||
            {};


        if (
            stats.hit_rate !==
            undefined
        ) {

            setText(
                "ctxHitRate",
                formatPercent(
                    stats.hit_rate
                )
            );
        }


        if (
            stats.total_requests !==
            undefined
        ) {

            setText(
                "ctxTotalRequests",
                formatNumber(
                    stats.total_requests
                )
            );
        }


        if (
            stats.cache_size !==
            undefined
        ) {

            setText(
                "ctxCacheSize",
                formatNumber(
                    stats.cache_size
                )
            );
        }


        console.log(
            "SmartEdge AI completed successfully."
        );


    } catch (error) {

        console.error(
            "SmartEdge AI error:",
            error
        );


        if (errorElement) {

            errorElement.textContent =
                error.message ||

                "Unable to connect to SmartEdge AI.";

            errorElement.style.display =
                "block";
        }


        if (answerElement) {

            answerElement.textContent =
                "Unable to generate an AI response.";
        }


    } finally {

        if (loadingElement) {

            loadingElement.style.display =
                "none";
        }


        if (button) {

            button.disabled =
                false;

            button.innerText =
                button.dataset.oldText ||

                "Ask SmartEdge";
        }
    }
}


// ============================================================
// INITIALIZATION
// ============================================================

async function initializeSmartEdge() {

    console.log(
        "SmartEdge frontend initialized."
    );


    initializeNavigation();


    // ========================================================
    // FIX: CONNECT ASK AI BUTTON
    // ========================================================

    const askButton =
        $("askAiBtn");


    if (askButton) {

        askButton.removeEventListener(
            "click",
            askSmartEdge
        );


        askButton.addEventListener(
            "click",
            askSmartEdge
        );

    } else {

        console.warn(
            "Ask AI button #askAiBtn was not found."
        );
    }


    await checkAPIConnection();


    await Promise.all([
        loadLiveStats(),
        loadBenchmark(),
        loadCacheContents()
    ]);


    // ========================================================
    // AUTO REFRESH
    // ========================================================

    if (refreshTimer) {

        clearInterval(
            refreshTimer
        );
    }


    refreshTimer =
        setInterval(
            async () => {

                const connected =
                    await checkAPIConnection();


                if (!connected) {
                    return;
                }


                await Promise.all([
                    loadLiveStats(),
                    loadCacheContents()
                ]);

            },

            5000
        );
}


// ============================================================
// GLOBAL FUNCTIONS
// ============================================================

window.runPrediction =
    runPrediction;


window.sendCacheRequest =
    sendCacheRequest;


window.resetCache =
    resetCache;


window.askSmartEdge =
    askSmartEdge;


window.loadLiveStats =
    loadLiveStats;


window.loadCacheContents =
    loadCacheContents;


window.loadBenchmark =
    loadBenchmark;


window.navigate =
    navigate;


window.showView =
    showView;


// ============================================================
// START
// ============================================================

if (
    document.readyState ===
    "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        initializeSmartEdge,
        {
            once:
                true
        }
    );

} else {

    initializeSmartEdge();
}