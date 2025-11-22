import httpx


async def get_top_works_raw_openalex(topic_id, k=20):
    # Last 10 years
    base_url = "https://api.openalex.org/works"
    params = {
        # Filter: Topic ID + Published from 2015-01-01
        "filter": f"primary_topic.id:{topic_id},from_publication_date:2015-01-01",
        "sort": "cited_by_count:desc",
        "per_page": k,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

    return data["results"]


async def get_topic_metadata(topic_id, k=20):
    works = await get_top_works_raw_openalex(topic_id, k=k)
    titles = [work["title"] for work in works]
    citation_counts_per_year_list = [work["counts_by_year"] for work in works]
    citation_counts = []
    for work_count_list in citation_counts_per_year_list:
        citation_counts.append(sum([e["cited_by_count"] for e in work_count_list]))
    return titles, citation_counts
