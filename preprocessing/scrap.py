import concurrent.futures
import threading

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# When an athlete has received at least one olympic medal
# it adds an extra table containing his medals
# We want to skip this table when scrapping the data
TO_SKIP = ["Medals", "Gold", "Silver", "Bronze", "Total"]


def get_data(k):
    """
    Scrap the data for the athlete with id k

    Args:
        k (int): the id of the athlete

    Returns:
        list: the list of the entries for the given athlete
    """
    # Initialization
    data = []
    athlete_name = None
    sex = None
    age = None
    height = None
    weight = None
    team = None
    noc = None
    games = None
    year = None
    birth_year = None
    season = None
    city = None
    sport = None
    event = None
    medal = None

    try:
        # Scrap the data
        url = f"https://www.olympedia.org/athletes/{k}"
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")
        else:
            # In case we get a bad response, either the page doesn't exist
            # or we get a time out, we return an empty list to avoid breaking
            # the methods that expect this one to return a list
            # TODO: check the number of missing IDs in the final dataset and
            # see if the pages indeed do not exist or if we should define a
            # retry behavior here to get the data
            return []

        # Extracting athlete's name
        # It's the main title of the page
        athlete_name = soup.find("h1").text.strip()

        # Extracting athlete's sex
        sex = soup.find("th", string="Sex").find_next("td").text.strip()

        # Extracting athlete's birth year
        # Depending on the pages, the birth year is defined differently
        # It can be:
        #   - Year
        #   - Month Year or (Month Year)
        #   - Year in Location
        #   - in Location => in that case we don't have a birth year
        #   - Month Year in Location
        #   - (Year or Year)
        #   - Day Month Year ...
        born = soup.find("th", string="Born")
        if born is not None:
            birth_date = born.find_next("td").text.strip()
            splits = birth_date.split(" ")
            if len(splits) == 1:
                birth_year = int(splits[0])
            elif len(splits) == 2:
                birth_year = int(splits[1].split(")")[0])
            elif splits[1] == "in":
                birth_year = int(splits[0])
            elif splits[0] == "in":
                pass
            elif splits[2] == "in":
                birth_year = int(splits[1])
            elif splits[1] == "or":
                birth_year = int(splits[0].split(")")[0])
            else:
                birth_year = int(splits[2])

        # Extracting athlete's height and weight
        measurements = soup.find("th", string="Measurements")
        # For some athletes we don't have any measurements, so the tag is not
        # even present on the page
        if measurements is not None:
            height_weight = measurements.find_next("td").text.strip().split("/")
            height = height_weight[0].strip()
            # In some cases, we only have the height
            if len(height_weight) > 1:
                weight = height_weight[1].strip()

        # country = soup.find("th", string="NOC").find_next("td").text.strip().split()[0]

        # Get the tables, it's where we find the data about the participations
        game_rows = soup.select("tbody > tr")
        for game_row in game_rows:
            all = game_row.find_all("td")

            # Remove any extra name that would come after a new line
            tmp = all[0].text.strip().split("\n")[0]

            # In case of a table to skip, go to the next one
            if tmp in TO_SKIP:
                continue

            # The first line of each table defines the games
            # The next lines define the actual results, and have a
            # first empty cell
            if tmp != "":
                games = tmp
                year, season = games.split(" ")[:2]
                year = int(year.split("-")[0])  # handle some rare formatting
                # Check if we have access to the birth year before calculating
                # the age of the athlete during the games
                if birth_year is not None:
                    age = year - birth_year
                # Get the sport and NOC, and then the following cells are empty
                # so go to the next line where we will find the results
                sport = all[1].text.strip()
                noc = all[2].text.strip()
                continue

            # Get the event, the team and the medal
            # We could also get the exact position on this website, but we don't
            # know the total number of participants on the event so it doesn't
            # provide much more information
            event = all[1].text.strip().split("\n")[0]
            team = all[2].text.strip()
            # pos = all[3].text.strip().split("=")[-1]
            medal = all[4].text.strip()

            # Add the current entry to the list of the athlete
            data.append(
                [
                    k,
                    athlete_name,
                    sex,
                    age,
                    height,
                    weight,
                    team,
                    noc,
                    games,
                    year,
                    season,
                    city,
                    sport,
                    event,
                    medal,
                ]
            )
    # In case of an exception, print the id and the error message
    # It makes it easier to understand what went wrong by checking
    # the webpage directly. It's mostly due to very rare formattings
    except Exception as e:
        print(f"Exception for k={k}: {e}")
    # Finally, return the collected data for the athlete
    return data


def run_batch(batch):
    """
    Scrap the data for the athletes between
    batch * 10,000 + 1 and (batch + 1) * 10,000

    Args:
        batch (int): the batch number
    """
    # Concurrent computing parameters
    num_workers = 12
    lock = threading.Lock()
    final = []
    batch_size = 10000

    start = 1 + batch * batch_size
    end = (batch + 1) * batch_size

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit scraping tasks
            future_to_url = {executor.submit(get_data, k): k for k in np.arange(start, end + 1)}
            for future in concurrent.futures.as_completed(future_to_url):
                data = future.result()
                print(".", end="")
                if data:
                    # In case of an empty list, print the id of the page that went wrong
                    if len(data) == 0:
                        print(f"Bad response: {future_to_url[future]}")
                    # Use a lock to safely append the resulting lines to the shared list
                    with lock:
                        for line in data:
                            final.append(line)

    # Handle when killing the running process
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)

    # Create a data frame to store the data
    df = pd.DataFrame(
        final,
        columns=[
            "ID",
            "Name",
            "Sex",
            "Age",
            "Height",
            "Weight",
            "Team",
            "NOC",
            "Games",
            "Year",
            "Season",
            "City",
            "Sport",
            "Event",
            "Medal",
        ],
    )

    # Save the data frame to a csv file
    df.to_csv(f"./Datasets/splits/olympics_athletes_{start}_{end}.csv", index=False)


if __name__ == "__main__":
    run_batch(batch=19)
