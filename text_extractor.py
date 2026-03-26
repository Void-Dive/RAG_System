import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Artificial_intelligence_in_video_games"
OUTPUT_FILE = "Selected_Document.txt"

def scrape_wikipedia_article():
    """
    Fetch the hardcoded Wikipedia page, extract paragraph text,
    save it to Selected_Document.txt, and return to text.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(URL, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")

            extracted_text = "\n\n".join(
                p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
            )

            with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
                file.write(extracted_text)

            print(f"Success! Text extracted and saved to {OUTPUT_FILE}")
            return extracted_text
        else:
            print(f"Failed to fetch page. HTTP status code: {response.status_code}")
            return ""

    except Exception as e:
        print(f"Error fetching or parsing the page: {e}")
        return ""
        
def main():
    scrape_wikipedia_article()

if __name__ == "__main__":
    main()