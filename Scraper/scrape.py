import requests
from bs4 import BeautifulSoup

def load_from_web(urls):
    documents = []

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            text = soup.get_text(separator="", strip=True)

            documents.append(
                {
                "source":url,
                "content":text,
                }
                )
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            
    return documents
    
        
