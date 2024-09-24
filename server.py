import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

reviews = pd.read_csv("data/reviews.csv").to_dict("records")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
ERROR_BAD_REQUEST = "Bad Request"
VALID_LOCATION = [
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona",
]


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(
        self, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            query_string = environ["QUERY_STRING"]
            query_params = parse_qs(query_string)
            filtered_data = reviews
            location = query_params.get("location")
            start_date = query_params.get("start_date")
            end_date = query_params.get("end_date")
            if location:
                filtered_data = (
                    data for data in filtered_data if data["Location"] == location[0]
                )

            if start_date:
                filtered_data = (
                    data
                    for data in filtered_data
                    if datetime.strptime(data["Timestamp"], TIMESTAMP_FORMAT)
                    >= datetime.strptime(start_date[0], "%Y-%m-%d")
                )

            if end_date:
                filtered_data = (
                    data
                    for data in filtered_data
                    if datetime.strptime(data["Timestamp"], TIMESTAMP_FORMAT)
                    <= datetime.strptime(end_date[0], "%Y-%m-%d")
                )

            response_data = []
            for data in filtered_data:
                data["sentiment"] = self.analyze_sentiment(data["ReviewBody"])
                response_data.append(data)

            sorted_list = sorted(
                response_data, key=lambda k: k["sentiment"]["compound"]
            )
            response_body = json.dumps(sorted_list, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ],
            )

            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            input_stream = environ["wsgi.input"]
            body = input_stream.read(int(environ["CONTENT_LENGTH"]))
            data = parse_qs(body.decode("utf-8"))

            location = data.get("Location", [None])[0]
            review_body = data.get("ReviewBody", [None])[0]

            response_code = "201 OK"
            if location not in VALID_LOCATION or not review_body:
                response_code = f"400 {ERROR_BAD_REQUEST}"

            data = {}
            data["Location"] = location
            data["ReviewBody"] = review_body
            data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data["ReviewId"] = str(uuid.uuid4())

            if response_code == "201 OK":
                reviews.append(data)

            response_body = json.dumps(data, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response(
                response_code,
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ],
            )

            return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get("PORT", 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
