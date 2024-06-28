import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import cv2
import numpy as np
import requests
from pydantic import BaseModel

"""

This is most of my code for the vision part of my bookshelf updates. 

Audible - 

    - Amazon's API library for audible was shut down years back so I made my own that scrapes it / avoids anti-bot. 
    - For security of my algorithm, I blurred that out of the code.

Spotify -
    - Spotify's API is quite nice and it is what I use often but it's rather long. It's just some modifications of their web api. 
    - https://developer.spotify.com/documentation/web-api

    It looks very similar to this: 

                const getAccessToken = async function (): Promise<string> {
                const storedToken = await redis.get<string>("access_token");
                if (storedToken) return storedToken;

                type TokenResponse = {
                    access_token: string;
                    expires_in: number;
                    refresh_token: string;
                };

                const { access_token: accessToken, expires_in: expiresIn }: TokenResponse = (await fetch(
                    `${TOKEN_URL}?${new URLSearchParams({
                    grant_type: "refresh_token",
                    refresh_token: refreshToken,
                    }).toString()}`,
                    {
                    method: "POST",
                    headers: {
                        Authorization: `Basic ${authorization}`,
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    },
                ).then((r) => r.json())) as TokenResponse;

                console.log(accessToken)
                await redis.set("access_token", accessToken, { ex: expiresIn });
                return accessToken;
                };

Vision - 
    - This is the rather complex part. The engineering utopistic plan was to setup a webcam / arduino / raspberry pi setup to connect to my computer at my 
        home in Rochester, MN to take a picture of my bookshelf every 24~ hours. This works quite nicely for now but has been running into some issues. 

Putting it all together, I just feed it to a llama v3 model on my base server for it to interpret and luckily, we can get it to return the new / missing / old 
titles that was different (thanks to a trusty .txt document holding it all together since MongoDB has my account on suspension!) -- but this scattered code can
give you some idea on how it is put together. Most of the code on my website looks a lot better than this but enjoy the scaffolding that I constructed to convey
the message! 
    
Input: 
    - SPOTIFY, AUDIBLE bookshelves (mainly whatevers new to reduce token size)
    - ARDUINO OUTPUT (cleaned)
    - other tokenizing info for context 
    - (optionally) my own inputs, very occassionally 

Output: 
    - a coherent list of my books 

With that output, I clean it with some json cleaners, and use a LLAMA v3-like LLM to find URLs for each of those respective titles, update the json, then I send that
to the site's frontend.

"""
class AudibleClient:
    def get_user_library(self) -> List[Dict[str, Any]]:
        return [{"title": f"Audiobook {i}", "author": f"Author {i}", "date_added": datetime.now() - timedelta(days=i)} for i in range(10)]

class SpotifyClient:
    def get_recently_played(self) -> List[Dict[str, Any]]:
        return [{"track": f"Song {i}", "artist": f"Artist {i}", "played_at": datetime.now() - timedelta(hours=i)} for i in range(5)]

# Computer Vision with OpenCV
class BookshelfScanner:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.last_scan = set()

    def capture_image(self) -> np.ndarray:
        ret, frame = self.camera.read()
        return frame

    def process_image(self, image: np.ndarray) -> List[str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        book_titles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 200:
                roi = gray[y:y+h, x:x+w]
                title = f"Book {len(book_titles) + 1}"
                book_titles.append(title)
        
        return book_titles

    def get_new_books(self) -> List[str]:
        current_books = set(self.process_image(self.capture_image()))
        new_books = current_books - self.last_scan
        self.last_scan = current_books
        return list(new_books)

# Data models
class MediaItem(BaseModel):
    title: str
    creator: str
    type: str
    timestamp: datetime

class DailyUpdate(BaseModel):
    date: datetime
    new_items: List[MediaItem]

# LLaMA v3 Client
class LLaMAv3Client:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def process_data(self, data: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
        payload = {
            "data": data,
            "prompt": prompt
        }
        response = requests.post(f"{self.server_url}/process", json=payload)
        if response.status_code == 200:
            return response.json()["new_items"]
        else:
            raise Exception(f"LLaMA v3 server error: {response.text}")

# Main application
class MediaTracker:
    def __init__(self):
        self.audible = AudibleClient()
        self.spotify = SpotifyClient()
        self.bookshelf_scanner = BookshelfScanner()
        self.llama_client = LLaMAv3Client("https://ajaymisra.com/llama")
        self.api_endpoint = "https://api.ajay.dog/media-updates"
        self.last_update_time = datetime.now() - timedelta(days=1)

    def get_new_audiobooks(self) -> List[MediaItem]:
        library = self.audible.get_user_library()
        return [
            MediaItem(
                title=book["title"],
                creator=book["author"],
                type="audiobook",
                timestamp=book["date_added"]
            )
            for book in library if book["date_added"] > self.last_update_time
        ]

    def get_new_music(self) -> List[MediaItem]:
        recent_tracks = self.spotify.get_recently_played()
        return [
            MediaItem(
                title=track["track"],
                creator=track["artist"],
                type="music",
                timestamp=track["played_at"]
            )
            for track in recent_tracks if track["played_at"] > self.last_update_time
        ]

    def get_new_physical_books(self) -> List[MediaItem]:
        new_books = self.bookshelf_scanner.get_new_books()
        return [
            MediaItem(
                title=title,
                creator="Unknown",
                type="physical_book",
                timestamp=datetime.now()
            )
            for title in new_books
        ]

    def collect_new_data(self) -> Dict[str, List[MediaItem]]:
        return {
            "audiobooks": self.get_new_audiobooks(),
            "music": self.get_new_music(),
            "physical_books": self.get_new_physical_books()
        }

    def process_with_llama(self, data: Dict[str, List[MediaItem]]) -> List[Dict[str, Any]]:
        prompt = """
        Given the list of new items in the user's media collection (audiobooks, music, and physical books),
        please identify and return only the new items that have been added since the last update.
        Do not make any recommendations or suggestions. Simply return the list of new items with their details.
        """
        return self.llama_client.process_data(data, prompt)

    def generate_daily_update(self) -> DailyUpdate:
        new_data = self.collect_new_data()
        processed_data = self.process_with_llama(new_data)
        
        new_items = [
            MediaItem(
                title=item["title"],
                creator=item["creator"],
                type=item["type"],
                timestamp=datetime.fromisoformat(item["timestamp"])
            )
            for item in processed_data
        ]
        
        update = DailyUpdate(
            date=datetime.now(),
            new_items=new_items
        )
        
        self.last_update_time = datetime.now()
        return update

    def send_update_to_api(self, update: DailyUpdate):
        response = requests.post(
            self.api_endpoint,
            json=update.dict(),
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 200:
            print(f"Failed to send update: {response.text}")

    def run_daily(self):
        while True:
            try:
                update = self.generate_daily_update()
                self.send_update_to_api(update)
                print(f"Daily update sent successfully. New items: {[item.title for item in update.new_items]}")
            except Exception as e:
                print(f"Error during daily update: {str(e)}")
            time.sleep(86400)  

if __name__ == "__main__":
    tracker = MediaTracker()
    tracker.run_daily()