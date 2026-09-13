from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import os
import joblib
import pandas as pd

from api.agent_api import router as agent_router


# ============================================================
# SMARTEDGE API
# ============================================================

app = FastAPI(
    title="SmartEdge API",
    description="Machine-Learning-Assisted Predictive Edge Caching API",
    version="3.1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"D:\smart edge")

FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_FILE = (
    BASE_DIR
    / "model"
    / "smartedge_random_forest.pkl"
)

DATASET_FILE = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "final_ml_dataset.csv"
)

RESULT_FILE = (
    BASE_DIR
    / "cache"
    / "results"
    / "final_cache_comparison.csv"
)


# ============================================================
# SMARTEDGE CONFIGURATION
# ============================================================

CACHE_CAPACITY = 100

# 0.50 = 50%
THRESHOLD = 0.50

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "past_frequency",
    "recency_seconds",
    "content_size"
]


# ============================================================
# MODEL
# ============================================================

_model = None


def load_model():

    global _model

    if _model is None:

        if not MODEL_FILE.exists():

            raise FileNotFoundError(
                f"Model not found: {MODEL_FILE}"
            )

        _model = joblib.load(
            MODEL_FILE
        )

        # Keep API inference lightweight.
        if hasattr(_model, "n_jobs"):

            _model.n_jobs = 1

    return _model


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):

    hour: float = Field(
        ...,
        ge=0,
        le=23
    )

    day_of_week: float = Field(
        ...,
        ge=0,
        le=6
    )

    past_frequency: float = Field(
        ...,
        ge=0
    )

    recency_seconds: float = Field(
        ...,
        ge=0
    )

    content_size: float = Field(
        ...,
        gt=0
    )


class ResourceRequest(BaseModel):

    url: str = Field(
        ...,
        min_length=1
    )

    content_size: float = Field(
        ...,
        gt=0
    )


# ============================================================
# LIVE CACHE
#
# SmartEdge:
# ML admission + LRU eviction
# ============================================================

cache = OrderedDict()

request_history = {}


# ============================================================
# LIVE STATISTICS
# ============================================================

stats = {

    "total_requests": 0,

    "hits": 0,

    "misses": 0,

    "admissions": 0,

    "rejections": 0,

    "evictions": 0,

    "ml_predictions": 0,

    "total_response_time_ms": 0.0
}


# ============================================================
# ROOT / FRONTEND
# ============================================================

@app.get(
    "/",
    include_in_schema=False
)
def frontend():

    index_file = (
        FRONTEND_DIR
        / "index.html"
    )

    if index_file.exists():

        return FileResponse(
            index_file
        )

    return {

        "status":
            "online",

        "project":
            "SmartEdge",

        "message":
            "Frontend index.html not found"
    }


# ============================================================
# API INFO
# ============================================================

@app.get("/api")
def api_info():

    return {

        "status":
            "online",

        "project":
            "SmartEdge",

        "version":
            "3.1.0",

        "model":
            "Binary Random Forest",

        "classes": {

            "0":
                "NO_REUSE",

            "1":
                "REUSE"
        },

        "threshold":
            THRESHOLD,

        "threshold_percent":
            THRESHOLD * 100,

        "cache_capacity":
            CACHE_CAPACITY
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            (
                "healthy"
                if MODEL_FILE.exists()
                else "degraded"
            ),

        "model":
            "Random Forest",

        "model_exists":
            MODEL_FILE.exists(),

        "dataset_exists":
            DATASET_FILE.exists(),

        "results_exists":
            RESULT_FILE.exists(),

        "frontend_exists":
            (
                FRONTEND_DIR
                / "index.html"
            ).exists(),

        "cache_capacity":
            CACHE_CAPACITY,

        "current_cache_size":
            len(cache),

        "threshold":
            THRESHOLD,

        "threshold_percent":
            THRESHOLD * 100
    }


# ============================================================
# FEATURE DATAFRAME
# ============================================================

def make_feature_dataframe(
    hour,
    day_of_week,
    past_frequency,
    recency_seconds,
    content_size
):

    return pd.DataFrame(
        [[
            float(hour),
            float(day_of_week),
            float(past_frequency),
            float(recency_seconds),
            float(content_size)
        ]],
        columns=FEATURE_COLUMNS
    )


# ============================================================
# ML PREDICTION
#
# ACTUAL MODEL:
#
# Class 0 = Not Popular
# Class 1 = Popular
#
# Therefore:
#
# reusable_probability =
# probability of class 1
# ============================================================

def predict_cache_value(
    hour,
    day_of_week,
    past_frequency,
    recency_seconds,
    content_size
):

    model = load_model()

    features = make_feature_dataframe(

        hour,

        day_of_week,

        past_frequency,

        recency_seconds,

        content_size
    )

    probabilities = model.predict_proba(
        features
    )[0]

    classes = [
        int(value)
        for value in model.classes_
    ]

    # --------------------------------------------------------
    # Probability of class 1
    # --------------------------------------------------------

    if 1 in classes:

        class_1_index = (
            classes.index(1)
        )

        reusable_probability = float(
            probabilities[class_1_index]
        )

    else:

        reusable_probability = 0.0


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        probabilities.argmax()
    )

    predicted_class = int(
        classes[predicted_index]
    )


    # --------------------------------------------------------
    # Binary label
    # --------------------------------------------------------

    if predicted_class == 1:

        predicted_label = "REUSE"

    else:

        predicted_label = "NO_REUSE"


    # --------------------------------------------------------
    # Expected cache value
    # --------------------------------------------------------

    expected_cache_value = (
        reusable_probability
    )


    # --------------------------------------------------------
    # Individual class probabilities
    # --------------------------------------------------------

    class_probabilities = {}

    for cls, probability in zip(
        classes,
        probabilities
    ):

        class_probabilities[
            str(cls)
        ] = round(
            float(probability),
            6
        )


    return {

        "predicted_class":
            predicted_class,

        "predicted_label":
            predicted_label,

        "reusable_probability":
            reusable_probability,

        "expected_cache_value":
            expected_cache_value,

        "class_probabilities":
            class_probabilities
    }


# ============================================================
# MANUAL ML PREDICTION
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    start_time = perf_counter()

    try:

        prediction = predict_cache_value(

            hour=request.hour,

            day_of_week=request.day_of_week,

            past_frequency=
                request.past_frequency,

            recency_seconds=
                request.recency_seconds,

            content_size=
                request.content_size
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )


    probability = float(
        prediction[
            "reusable_probability"
        ]
    )


    # ========================================================
    # SINGLE SOURCE OF TRUTH
    # ========================================================

    should_cache = (
        probability >= THRESHOLD
    )


    # The displayed prediction must agree
    # with the admission threshold.

    if should_cache:

        predicted_label = "REUSE"

    else:

        predicted_label = "NO_REUSE"


    elapsed_ms = (
        perf_counter()
        -
        start_time
    ) * 1000


    return {

        "status":
            "success",

        "predicted_class":
            prediction[
                "predicted_class"
            ],

        "predicted_label":
            predicted_label,

        "reusable_probability":
            round(
                probability,
                6
            ),

        "reusable_probability_percent":
            round(
                probability * 100,
                2
            ),

        "expected_cache_value":
            round(
                probability,
                6
            ),

        "threshold":
            THRESHOLD,

        "threshold_percent":
            THRESHOLD * 100,

        "cache_decision":
            (
                "CACHE"
                if should_cache
                else "DO_NOT_CACHE"
            ),

        "decision_text":
            (
                "CACHE"
                if should_cache
                else "DO NOT CACHE"
            ),

        "class_probabilities":
            prediction[
                "class_probabilities"
            ],

        "response_time_ms":
            round(
                elapsed_ms,
                3
            )
    }


# ============================================================
# CACHE STATISTICS
# ============================================================

@app.get("/cache/stats")
def cache_stats():

    total = stats[
        "total_requests"
    ]

    hits = stats[
        "hits"
    ]

    misses = stats[
        "misses"
    ]


    if total > 0:

        hit_rate = (
            hits / total
        ) * 100

        miss_rate = (
            misses / total
        ) * 100

    else:

        hit_rate = 0.0

        miss_rate = 0.0


    average_response_time = (

        stats[
            "total_response_time_ms"
        ]

        / total

        if total > 0

        else 0.0
    )


    return {

        "status":
            "success",

        "total_requests":
            total,

        "hits":
            hits,

        "misses":
            misses,

        "hit_rate":
            round(
                hit_rate,
                2
            ),

        "miss_rate":
            round(
                miss_rate,
                2
            ),

        "admissions":
            stats[
                "admissions"
            ],

        "rejections":
            stats[
                "rejections"
            ],

        "evictions":
            stats[
                "evictions"
            ],

        "ml_predictions":
            stats[
                "ml_predictions"
            ],

        "cache_capacity":
            CACHE_CAPACITY,

        "cache_size":
            len(cache),

        "cache_utilization":
            round(
                (
                    len(cache)
                    /
                    CACHE_CAPACITY
                ) * 100,
                2
            ),

        "average_response_time_ms":
            round(
                average_response_time,
                3
            ),

        "threshold":
            THRESHOLD,

        "threshold_percent":
            THRESHOLD * 100
    }


# ============================================================
# CACHE CONTENTS
# ============================================================

@app.get("/cache")
def get_cache():

    items = []


    for url, item in cache.items():

        cached_at = item.get(
            "cached_at"
        )


        items.append({

            "url":
                url,

            "frequency":
                item[
                    "frequency"
                ],

            "predicted_class":
                item[
                    "predicted_class"
                ],

            "predicted_label":
                item[
                    "predicted_label"
                ],

            "reusable_probability":
                round(
                    item[
                        "reusable_probability"
                    ],
                    6
                ),

            "reusable_probability_percent":
                round(
                    item[
                        "reusable_probability"
                    ] * 100,
                    2
                ),

            "expected_cache_value":
                round(
                    item[
                        "expected_cache_value"
                    ],
                    6
                ),

            "content_size":
                item[
                    "content_size"
                ],

            "content_size_kb":
                round(
                    item[
                        "content_size"
                    ] / 1024,
                    3
                ),

            "cached_at":
                (
                    cached_at.isoformat()
                    if cached_at
                    else None
                )
        })


    return {

        "status":
            "success",

        "size":
            len(items),

        "capacity":
            CACHE_CAPACITY,

        "items":
            items
    }


# ============================================================
# REMOVE CACHE ITEM
# ============================================================

@app.delete("/cache/item")
def remove_cache_item(
    url: str
):

    if url not in cache:

        raise HTTPException(
            status_code=404,
            detail=f"Cache item not found: {url}"
        )


    del cache[url]


    return {

        "status":
            "success",

        "message":
            f"Removed {url} from cache",

        "cache_size":
            len(cache)
    }


# ============================================================
# RESET CACHE
# ============================================================

@app.post("/cache/reset")
def reset_cache():

    cache.clear()

    request_history.clear()


    for key in stats:

        stats[key] = 0


    return {

        "status":
            "success",

        "message":
            "SmartEdge cache and statistics reset",

        "cache_size":
            0
    }


# ============================================================
# SMARTEDGE LIVE REQUEST
# ============================================================

@app.post("/request")
def smartedge_request(
    request: ResourceRequest
):

    start_time = perf_counter()

    url = request.url.strip()

    content_size = float(
        request.content_size
    )


    if not url:

        raise HTTPException(
            status_code=422,
            detail="Resource URL cannot be empty."
        )


    now = datetime.now()


    current_index = stats[
        "total_requests"
    ]


    stats[
        "total_requests"
    ] += 1


    # ========================================================
    # CACHE HIT
    # ========================================================

    if url in cache:

        stats[
            "hits"
        ] += 1


        item = cache[url]


        item[
            "frequency"
        ] += 1


        item[
            "last_seen"
        ] = current_index


        item[
            "cached_at"
        ] = now


        # LRU
        cache.move_to_end(
            url
        )


        request_history[url] = {

            "frequency":
                item[
                    "frequency"
                ],

            "last_seen_time":
                now
        }


        elapsed_ms = (
            perf_counter()
            -
            start_time
        ) * 1000


        stats[
            "total_response_time_ms"
        ] += elapsed_ms


        probability = float(
            item[
                "reusable_probability"
            ]
        )


        return {

            "status":
                "success",

            "cache_status":
                "HIT",

            "message":
                "Resource served from SmartEdge edge cache.",

            "url":
                url,

            "content_size":
                content_size,

            "ml_prediction": {

                "predicted_class":
                    item[
                        "predicted_class"
                    ],

                "predicted_label":
                    item[
                        "predicted_label"
                    ],

                "reusable_probability":
                    round(
                        probability,
                        6
                    ),

                "reusable_probability_percent":
                    round(
                        probability * 100,
                        2
                    ),

                "expected_cache_value":
                    round(
                        item[
                            "expected_cache_value"
                        ],
                        6
                    )
            },

            "cache_decision": {

                "threshold":
                    THRESHOLD,

                "threshold_percent":
                    THRESHOLD * 100,

                "admitted":
                    True,

                "decision":
                    "CACHE",

                "decision_text":
                    "ADMITTED",

                "evicted_url":
                    None
            },

            "response_time_ms":
                round(
                    elapsed_ms,
                    3
                ),

            "cache_size":
                len(cache),

            "cache_capacity":
                CACHE_CAPACITY
        }


    # ========================================================
    # CACHE MISS
    # ========================================================

    stats[
        "misses"
    ] += 1


    previous = request_history.get(
        url
    )


    if previous is None:

        past_frequency = 0.0

        recency_seconds = (
            1_000_000.0
        )

    else:

        past_frequency = float(
            previous[
                "frequency"
            ]
        )

        recency_seconds = max(

            (
                now
                -
                previous[
                    "last_seen_time"
                ]
            ).total_seconds(),

            0.0
        )


    # ========================================================
    # ML PREDICTION
    # ========================================================

    try:

        prediction = predict_cache_value(

            hour=float(
                now.hour
            ),

            day_of_week=float(
                now.weekday()
            ),

            past_frequency=
                past_frequency,

            recency_seconds=
                recency_seconds,

            content_size=
                content_size
        )


        stats[
            "ml_predictions"
        ] += 1


    except Exception as error:

        elapsed_ms = (
            perf_counter()
            -
            start_time
        ) * 1000


        stats[
            "total_response_time_ms"
        ] += elapsed_ms


        raise HTTPException(
            status_code=500,
            detail=f"ML prediction failed: {error}"
        )


    probability = float(
        prediction[
            "reusable_probability"
        ]
    )


    # ========================================================
    # SINGLE ADMISSION RULE
    # ========================================================

    should_cache = (
        probability >= THRESHOLD
    )


    predicted_class = int(
        prediction[
            "predicted_class"
        ]
    )


    predicted_label = (

        "REUSE"
        if should_cache
        else "NO_REUSE"
    )


    evicted_url = None


    # ========================================================
    # ADMIT TO CACHE
    # ========================================================

    if should_cache:

        # ----------------------------------------------------
        # Cache full -> evict LRU
        # ----------------------------------------------------

        if len(cache) >= CACHE_CAPACITY:

            evicted_url, _ = (
                cache.popitem(
                    last=False
                )
            )

            stats[
                "evictions"
            ] += 1


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        cache[url] = {

            "frequency":
                1,

            "last_seen":
                current_index,

            "last_seen_time":
                now,

            "cached_at":
                now,

            "predicted_class":
                predicted_class,

            "predicted_label":
                predicted_label,

            "reusable_probability":
                probability,

            "expected_cache_value":
                probability,

            "content_size":
                content_size
        }


        stats[
            "admissions"
        ] += 1


        decision_text = (
            "ADMITTED"
        )


    # ========================================================
    # REJECT
    # ========================================================

    else:

        stats[
            "rejections"
        ] += 1


        decision_text = (
            "REJECTED"
        )


    # ========================================================
    # UPDATE REQUEST HISTORY
    # ========================================================

    request_history[url] = {

        "frequency":
            (
                previous[
                    "frequency"
                ] + 1

                if previous

                else 1
            ),

        "last_seen_time":
            now
    }


    elapsed_ms = (
        perf_counter()
        -
        start_time
    ) * 1000


    stats[
        "total_response_time_ms"
    ] += elapsed_ms


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status":
            "success",

        "cache_status":
            "MISS",

        "message":
            (
                "Resource admitted into SmartEdge edge cache."
                if should_cache
                else
                "Resource was rejected and NOT stored because its ML reuse probability is below 50%."
            ),

        "url":
            url,

        "content_size":
            content_size,

        "ml_prediction": {

            "predicted_class":
                predicted_class,

            "predicted_label":
                predicted_label,

            "reusable_probability":
                round(
                    probability,
                    6
                ),

            "reusable_probability_percent":
                round(
                    probability * 100,
                    2
                ),

            "expected_cache_value":
                round(
                    probability,
                    6
                )
        },

        "cache_decision": {

            "threshold":
                THRESHOLD,

            "threshold_percent":
                THRESHOLD * 100,

            "admitted":
                should_cache,

            "decision":
                (
                    "CACHE"
                    if should_cache
                    else "DO_NOT_CACHE"
                ),

            "decision_text":
                decision_text,

            "evicted_url":
                evicted_url
        },

        "response_time_ms":
            round(
                elapsed_ms,
                3
            ),

        "cache_size":
            len(cache),

        "cache_capacity":
            CACHE_CAPACITY
    }


# ============================================================
# OFFLINE BENCHMARK RESULTS
#
# This reads your existing CSV.
# It DOES NOT rerun the experiment.
# ============================================================

@app.get("/summary")
def summary():

    if not RESULT_FILE.exists():

        return {

            "status":
                "error",

            "message":
                f"Benchmark results not found: {RESULT_FILE}"
        }


    try:

        df = pd.read_csv(
            RESULT_FILE
        )

    except Exception as error:

        return {

            "status":
                "error",

            "message":
                f"Could not read benchmark results: {error}"
        }


    algorithms = {}


    for _, row in df.iterrows():

        algorithm = str(
            row[
                "algorithm"
            ]
        )


        hits = int(
            row[
                "hits"
            ]
        )


        misses = int(
            row[
                "misses"
            ]
        )


        total = (
            hits + misses
        )


        if total > 0:

            hit_rate = (
                hits / total
            )

            miss_rate = (
                misses / total
            )

        else:

            hit_rate = 0.0
            miss_rate = 0.0


        algorithms[
            algorithm
        ] = {

            "requests":
                total,

            "hits":
                hits,

            "misses":
                misses,

            # Decimal form.
            "hit_rate":
                hit_rate,

            "miss_rate":
                miss_rate,

            # Percentage form.
            "hit_rate_percent":
                round(
                    hit_rate * 100,
                    3
                ),

            "miss_rate_percent":
                round(
                    miss_rate * 100,
                    3
                ),

            "evictions":
                int(
                    row.get(
                        "evictions",
                        0
                    )
                ),

            "proactive_insertions":
                int(
                    row.get(
                        "proactive_insertions",
                        0
                    )
                ),

            "origin_requests":
                int(
                    row.get(
                        "origin_requests",
                        misses
                    )
                ),

            "average_latency_ms":
                float(
                    row.get(
                        "average_latency_ms",
                        0
                    )
                ),

            "hit_latency_ms":
                float(
                    row.get(
                        "hit_latency_ms",
                        0
                    )
                ),

            "miss_latency_ms":
                float(
                    row.get(
                        "miss_latency_ms",
                        0
                    )
                ),

            "bandwidth_mb":
                float(
                    row.get(
                        "bandwidth_mb",
                        0
                    )
                ),

            "cache_utilization_percent":
                float(
                    row.get(
                        "cache_utilization_percent",
                        0
                    )
                ),

            "throughput_req_per_sec":
                float(
                    row.get(
                        "throughput_req_per_sec",
                        0
                    )
                )
        }


    return {

        "status":
            "success",

        "simulation_requests":
            max(
                [
                    item[
                        "requests"
                    ]
                    for item in algorithms.values()
                ],
                default=0
            ),

        "algorithms":
            algorithms
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print()
    print("=" * 70)
    print("SMARTEDGE 3.1 API")
    print("=" * 70)

    print(
        "Model:",
        MODEL_FILE
    )

    print(
        "Model exists:",
        MODEL_FILE.exists()
    )

    print(
        "Dataset exists:",
        DATASET_FILE.exists()
    )

    print(
        "Benchmark exists:",
        RESULT_FILE.exists()
    )

    print(
        "Frontend exists:",
        (
            FRONTEND_DIR
            / "index.html"
        ).exists()
    )

    print(
        "Cache capacity:",
        CACHE_CAPACITY
    )

    print(
        "Admission threshold:",
        f"{THRESHOLD * 100:.0f}%"
    )

    print(
        "Model classes:"
        " 0=NO_REUSE, 1=REUSE"
    )

    print("=" * 70)
    print()


app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND_DIR,
        html=True
    ),
    name="frontend-assets"
)