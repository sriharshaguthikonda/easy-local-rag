import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from Semantic_chunking import split_into_chunks  # Import chunking function


async def fetch_and_chunk(crawler, url):
    try:
        result = await crawler.arun(url=url)
        if result.success:
            soup = BeautifulSoup(result.html, "html.parser")
            text = soup.get_text()
            chunks = split_into_chunks(text, generate_id=False)
            # Use chunking function from semantic_chunking.py  here the id is not generated
            return chunks
        else:
            print(f"Error fetching {url}: {result.error_message}")
            return []
    except Exception as e:
        print(f"An error occurred while fetching {url}: {e}")
        return []


async def get_web_context(query, num_results=5):
    search_url = f"https://www.google.com/search?q={query}&num={num_results}"
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=search_url)
            if result.success:
                soup = BeautifulSoup(result.html, "html.parser")
                results = []
                for g in soup.find_all("div", class_="tF2Cxc"):
                    title = g.find("h3").get_text() if g.find("h3") else "No title"
                    link = g.find("a")["href"] if g.find("a") else "No link"
                    snippet = (
                        g.find("div", class_="IsZvec").get_text()
                        if g.find("div", class_="IsZvec")
                        else "No snippet"
                    )
                    results.append({"title": title, "link": link, "snippet": snippet})
                for i, result in enumerate(results, start=1):
                    print(f"Result {i}:")
                    print(f"Title: {result['title']}")
                    print(f"Link: {result['link']}")
                    print(f"Snippet: {result['snippet']}\n")
                    if result["link"] != "No link":
                        chunks = await fetch_and_chunk(crawler, result["link"])
                        for j, chunk in enumerate(chunks, start=1):
                            print(f"Chunk {j}:\n{chunk}\n")
            else:
                print(f"Error: {result.error_message}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(get_web_context("python programming", num_results=2))
