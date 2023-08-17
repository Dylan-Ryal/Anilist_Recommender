import requests
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from IPython.core.display import HTML

class DataHandler:
    def __init__(self, username):
        self.genres = {}
        self.tags = {}
        self.studios = {}
        self.all_staff = {}
        self.username = username

    def fetch_user_lists(self):
        query = """
            query($name: String!){
                MediaListCollection(userName: $name, type: ANIME){
                    lists{
                        name
                        entries{
                            ... mediaListEntry
                        }
                    }
                }
            }

            fragment mediaListEntry on MediaList{
                media{
                    title{romaji}
                    tags{
                        name
                        rank
                    }
                    genres
                    averageScore
                    studios(isMain: true){
                        nodes{
                            name
                            isAnimationStudio
                        }
                    }
                    staff(sort: [RELEVANCE]) {
                        edges {
                            role
                            node {
                                id
                            }
                        }
                    }
                    coverImage {large}
                    siteUrl
                }
                scoreRaw: score(format: POINT_100)
            }
        """

        variables = {
            "name": self.username
        }

        url = 'https://graphql.anilist.co'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        data = {
            "query": query,
            "variables": variables
        }

        response = requests.post(url, json=data, headers=headers)

        def handleResponse(response):
            json_data = response.json()
            return json_data if response.ok else json_data

        def handleData(data):
            return data['data']['MediaListCollection']['lists']

        def handleError(error):
            print("Error:", error)

        try:
            response = requests.post(url, json=data, headers=headers)
            response_data = handleResponse(response)
            user_lists = handleData(response_data)
        except Exception as e:
            handleError(e)
        
        return user_lists
    
    def fetch_season_list(self, year, season):
        query = '''
            query($season: MediaSeason, $year: Int){
                Page(page: 1) {
                media(season: $season, seasonYear: $year, format: TV, sort: [SCORE_DESC]) {
                    title {
                    romaji
                    }
                    tags {
                    name
                    rank
                    }
                    genres
                    averageScore
                    studios(isMain: true) {
                    nodes {
                        name
                        isAnimationStudio
                    }
                    }
                    staff(sort: [RELEVANCE]) {
                    edges {
                        role
                        node {
                        id
                        }
                    }
                    }
                    coverImage {
                    large
                    }
                    siteUrl
                }
                }
            }   
        '''

        variables = {
            "year": year,
            "season": season
        }

        url = 'https://graphql.anilist.co'
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        data = {
            "query": query,
            "variables": variables
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            json_data = response.json()
            if 'data' in json_data:
                season_list = json_data['data']['Page']
            else:
                print("No data found in the response.")
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)
        
        return season_list
    
    def create_data_array(self, input_list, is_media_list):
        data = []
        default_genre_value = self.default_values['defaultGenreValue']
        default_tag_value = self.default_values['defaultTagValue']
        default_studio_value = self.default_values['defaultStudioValue']
        default_staff_value = self.default_values['defaultStaffValue']

        for entry in input_list:
            input_data = []
            additional_data = {}

            genre_scores = []
            genres = entry['genres'] if is_media_list else entry['media']['genres']
            for genre in genres:
                genre_scores.append(self.genres[genre] if genre in self.genres else default_genre_value)
            if len(genre_scores) != 0:
                input_data.append(np.mean(genre_scores))
            else:
                input_data.append(default_genre_value)

            studio_scores = []
            studios = entry['studios']['nodes'] if is_media_list else entry['media']['studios']['nodes']
            ani_studios = [studio for studio in studios if studio['isAnimationStudio']]
            for studio in ani_studios:
                studio_scores.append(self.studios[studio['name']] if studio['name'] in self.studios else default_studio_value)
            if len(studio_scores) != 0:
                input_data.append(np.mean(studio_scores))
            else:
                input_data.append(default_studio_value)

            tag_scores = []
            tags = entry['tags'] if is_media_list else entry['media']['tags']
            top_tags = [tag for tag in tags if tag['rank'] >= 0]
            total_rank = 0
            for tag in top_tags:
                    tag_scores.append(self.tags[tag['name']] * tag['rank'] if tag['name'] in self.tags else default_tag_value)
                    total_rank += tag["rank"]
            if len(tag_scores) != 0:
                input_data.append(np.sum(tag_scores) / total_rank)
            else:
                input_data.append(default_tag_value)

            staff_scores = []
            all_staff = entry['staff']['edges'] if is_media_list else entry['media']['staff']['edges']
            for staff in all_staff:
                if staff["role"] in ["Director", "Series Composition", "Character Design", "Music", "Art Director"]:
                    staff_id = staff['node']['id']
                    staff_scores.append(self.all_staff[staff_id] if staff_id in self.all_staff else default_staff_value)
            if len(staff_scores) != 0:
                input_data.append(np.mean(staff_scores))
            else:
                input_data.append(default_staff_value)

            input_data.append(entry['averageScore'] if is_media_list else entry['media']['averageScore'])
            input_data.append(0 if is_media_list else entry['scoreRaw'])

            additional_data['title'] = entry['title']['romaji'] if is_media_list else entry['media']['title']['romaji']
            additional_data['image'] = entry['coverImage']['large'] if is_media_list else entry['media']['coverImage']['large']
            additional_data['url'] = entry['siteUrl'] if is_media_list else entry['media']['siteUrl']

            data.append({'inputData': input_data, 'additionalData': additional_data})

        return data
    
    def count_data(self, entry_list):
        for entry in entry_list:
            for genre in entry['media']['genres']:
                if genre in self.genres:
                    self.genres[genre]['count'] += 1
                    self.genres[genre]['total'] += entry['scoreRaw']
                else:
                    self.genres[genre] = {'count': 1, 'total': entry['scoreRaw']}

            top_tags = [tag for tag in entry['media']['tags'] if tag['rank'] >= 0]
            for tag in top_tags:
                    if tag['name'] in self.tags:
                        self.tags[tag['name']]['rank'] += tag["rank"]
                        self.tags[tag['name']]['total'] += entry['scoreRaw'] * tag["rank"]
                    else:
                        self.tags[tag['name']] = {'rank': tag["rank"], 'total': entry['scoreRaw'] * tag["rank"]}

            ani_studios = [studio for studio in entry['media']['studios']['nodes'] if studio['isAnimationStudio']]
            for studio in ani_studios:
                if studio['name'] in self.studios:
                    self.studios[studio['name']]['count'] += 1
                    self.studios[studio['name']]['total'] += entry['scoreRaw']
                else:
                    self.studios[studio['name']] = {'count': 1, 'total': entry['scoreRaw']}

            for staff in entry['media']['staff']['edges']:
                if staff["role"] in ["Director", "Series Composition", "Character Design", "Music", "Art Director"]:
                    staff_id = staff['node']['id']
                    if staff_id in self.all_staff:
                        self.all_staff[staff_id]['count'] += 1
                        self.all_staff[staff_id]['total'] += entry['scoreRaw']
                    else:
                        self.all_staff[staff_id] = {'count': 1, 'total': entry['scoreRaw']}
    
    def average_data(self):
        genre_values = []
        for key, value in self.genres.items():
            self.genres[key] = value['total'] / value['count']
            genre_values.append(self.genres[key])
        default_genre_value = np.mean(genre_values)

        tag_values = []
        for key, value in self.tags.items():
            self.tags[key] = value['total'] / value['rank']
            tag_values.append(self.tags[key])
        default_tag_value = np.mean(tag_values)

        studio_values = []
        for key, value in self.studios.items():
            self.studios[key] = value['total'] / value['count']
            studio_values.append(self.studios[key])
        default_studio_value = np.mean(studio_values)

        staff_values = []
        for key, value in self.all_staff.items():
            self.all_staff[key] = value['total'] / value['count']
            staff_values.append(self.all_staff[key])
        default_staff_value = np.mean(staff_values)

        self.default_values = {
            'defaultGenreValue': default_genre_value,
            'defaultTagValue': default_tag_value,
            'defaultStudioValue': default_studio_value,
            'defaultStaffValue': default_staff_value
        }


username = input("Enter a username: ")

valid_year = False
while valid_year == False:
    year = input("Enter a year: ")
    if year.isdigit() == False:
        print("Invalid Year: Not an Integer")
        continue
    elif int(year) < 1940:
        print("Invalid Year: Out of Range")
        continue
    valid_year = True

valid_seasons = ["WINTER", "SPRING", "SUMMER", "FALL"]
valid_season = False
while valid_season == False:
    season = input("Enter a season (Winter, Spring, Summer, Fall or All): ").upper()
    if season not in valid_seasons and season != "ALL":
        print("Invalid Season: Please enter either Winter, Spring, Summer, Fall or All")
        continue
    valid_season = True

valid_include = False
while valid_include == False:
    include_dropped = input("Include dropped shows? Enter Yes or No: ").upper()
    if include_dropped != "YES" and include_dropped != "NO":
        print("Invalide Input: Please enter Yes or No")
        continue
    valid_include = True

print("Processing User Data...")
user_data_handler = DataHandler(username)

user_lists = user_data_handler.fetch_user_lists()
completed_list = filter(lambda list_: list_["name"] == "Completed", user_lists)
completed_list = list(completed_list)[0]

user_data_handler.count_data(completed_list["entries"])
user_data_handler.average_data()
data = user_data_handler.create_data_array(completed_list["entries"], False)

data_array = [entry["inputData"] for entry in data]
user_df = pd.DataFrame(data_array, columns=["genre", "studio", "tags", "staff", "community_score", "user_score"])
user_df = user_df.query("user_score != 0")
user_df["community_score"].fillna(65, inplace=True)

x = user_df.drop(["user_score"], axis=1)
y = user_df["user_score"]

print("Training Model...")
classifier = LinearRegression(fit_intercept=True)
classifier.fit(x, y)

def season_dataframe(season):
    print("Processing Season Data...")
    season_list = user_data_handler.fetch_season_list(year, season)
    season_data = user_data_handler.create_data_array(season_list["media"], True)
    season_data_array = [entry["inputData"] for entry in season_data]
    season_df = pd.DataFrame(season_data_array, columns=["genre", "studio", "tags", "staff", "community_score", "user_score"])
    season_df["community_score"].fillna(65, inplace=True)
    season_info_df = pd.DataFrame([[entry["additionalData"]["title"], entry["additionalData"]["image"]] for entry in season_data], columns=["title", "image"])
    return season_df, season_info_df

season_df = pd.DataFrame(columns=["genre", "studio", "tags", "staff", "community_score", "user_score"])
season_info_df = pd.DataFrame(columns=["title", "image"])
if season == "ALL":
    for a_season in valid_seasons:
        a_season_df, a_season_info_df = season_dataframe(a_season)
        season_df = pd.concat([season_df, a_season_df], axis=0)
        season_info_df = pd.concat([season_info_df, a_season_info_df], axis=0)
    season_df.reset_index(drop=True, inplace=True)
    season_info_df.reset_index(drop=True, inplace=True)
else:
    season_df, season_info_df = season_dataframe(season)

print("Making Predictions...")
season_predictions = classifier.predict(season_df.drop(["user_score"], axis=1))
season_predictions_df = pd.DataFrame(np.round(season_predictions), columns=["score"])
season_predictions_df = pd.concat([season_info_df, season_predictions_df], join="inner", axis=1)
season_predictions_df["community"] = season_df["community_score"]
season_predictions_df = season_predictions_df.sort_values(by=["score"], ascending=False)
y_std = np.std(y)
season_predictions_df["score"] = ((season_predictions_df["score"] - np.mean(y) + y_std) / (2 * y_std)) * 100
season_predictions_df = season_predictions_df[season_predictions_df["community"] >= 65]
season_predictions_df["score"] = season_predictions_df["score"].apply(lambda x: x if x <= 100.0 else 100.0)
season_predictions_df["score"] = season_predictions_df["score"].apply(lambda x: x if x >= 0.0 else 0.0)
season_predictions_df["score"] = season_predictions_df["score"].apply(lambda x: str(int(x)) + "%")

    
completed_titles = [entry["media"]["title"]["romaji"] for entry in completed_list["entries"]]
if include_dropped == "YES":
    season_predictions_df = season_predictions_df[(~season_predictions_df["title"].isin(completed_titles))]
else:
    dropped_list = filter(lambda list_: list_["name"] == "Dropped", user_lists)
    dropped_list = list(dropped_list)[0]
    dropped_titles = [entry["media"]["title"]["romaji"] for entry in dropped_list["entries"]]
    season_predictions_df = season_predictions_df[(~season_predictions_df["title"].isin(completed_titles)) & (~season_predictions_df["title"].isin(dropped_titles))]

season_predictions_df.reset_index(drop=True, inplace=True)
season_predictions_df.index += 1

def path_to_image_html(path):
    return '<img src="'+ path + '" width="60" >'

pd.set_option('display.max_colwidth', None)

image_cols = ['image']  

format_dict = {}
for image_col in image_cols:
    format_dict[image_col] = path_to_image_html
html = HTML(season_predictions_df.to_html(escape=False ,formatters=format_dict))
if not os.path.exists("Predictions"):
    os.mkdir("Predictions")
html_file = ".\\Predictions\\" + username + "_" + year + "_" + season + ".html"
with open(html_file, 'w', encoding="utf-8") as f:
    f.write(html.data)
    print(f"HTML File Generated! {html_file}")