import requests


API_URL = "http://127.0.0.1:8001"


def get_cache_stats():

    response = requests.get(
        f"{API_URL}/cache/stats",
        timeout=5
    )

    response.raise_for_status()

    return response.json()


def get_cache_state():

    response = requests.get(
        f"{API_URL}/cache",
        timeout=5
    )

    response.raise_for_status()

    return response.json()


def get_health():

    response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    response.raise_for_status()

    return response.json()


def get_live_context():

    context = {}

    try:
        context["health"] = get_health()
    except Exception as e:
        context["health_error"] = str(e)

    try:
        context["cache_stats"] = get_cache_stats()
    except Exception as e:
        context["cache_stats_error"] = str(e)

    try:
        context["cache_state"] = get_cache_state()
    except Exception as e:
        context["cache_state_error"] = str(e)

    return context


if __name__ == "__main__":

    print("\nSMARTEDGE LIVE TOOLS")
    print("====================")

    context = get_live_context()

    print("\nHealth:")
    print(context.get("health"))

    print("\nCache Stats:")
    print(context.get("cache_stats"))

    print("\nCache State:")
    print(context.get("cache_state"))