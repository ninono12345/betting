# main_code.py

import requests
import json
import time
import threading
import pickle
import re
import pandas as pd

import re
from bs4 import BeautifulSoup
from collections import defaultdict

def parse_match_events_from_html(html_content):
    """
    Parses the HTML content of a match timeline table to extract structured event data.

    This function identifies key events like goals, corners, and cards, and
    extracts the time and the team responsible. It is designed to ignore informational
    lines like "Race to X Corners" and disallowed goals that are not singular,
    timestamped actions.

    Args:
        html_content (str): The HTML string containing the timeline data.

    Returns:
        list: A list of dictionaries, where each dictionary represents an event
              and contains 'time', 'event_type', and 'team'. Returns an empty
              list if the input is invalid.
    """
    if not html_content:
        return []

    parsed_events = []
    soup = BeautifulSoup(html_content, 'html.parser')
    event_list = soup.find('ul', id='race_events')

    if not event_list:
        return []

    for item in event_list.find_all('li', class_='bullet-item'):
        text = item.get_text(strip=True)

        # Skip informational lines that don't represent a direct team action
        if 'Race to' in text or 'Injury Time' in text or 'Score After' in text:
            continue

        time_match = re.match(r"([\d\+]+)'", text)
        if not time_match:
            continue
        time = time_match.group(1)

        event_type = None
        # MODIFIED: Check for "Goal" but exclude if "Disallowed Goal" is present
        if 'Goal' in text and 'Disallowed Goal' not in text:
            event_type = 'goal'
        elif 'Corner' in text:
            event_type = 'corner'
        elif 'Yellow Card' in text:
            event_type = 'yellow_card'
        elif 'Red Card' in text:
            event_type = 'red_card'

        if not event_type:
            continue

        team = None
        # Try to find team in parentheses first, e.g., (CA Colegiales)
        team_match = re.search(r'\((.*?)\)', text)
        if team_match:
            team = team_match.group(1).replace('~', '').strip()
        else:
            # If not in parentheses, split by '-' and take the last part
            parts = text.split('-')
            if len(parts) > 1:
                team = parts[-1].strip()
                # Handle cases like 'Penalty' by ensuring it's not a generic word
                if team.lower() in ['penalty']:
                    team = None # Or find a better way if team is elsewhere

        # Final cleanup for the team name
        if team:
            team = team.replace('~', '').strip()

        if team:
            parsed_events.append({
                'time': time,
                'event_type': event_type,
                'team': team
            })

    return parsed_events


def create_timeline_from_parsed_events(events):
    """
    Organizes a list of parsed events into a structured timeline.

    The timeline groups events by action type (goals, corners, etc.) and then
    by the team that performed the action. Event times are converted to integers
    and sorted.

    Args:
        events (list): A list of event dictionaries, typically from the
                       parse_match_events_from_html function.

    Returns:
        dict: A dictionary where keys are action types (e.g., 'goals').
              Each action type contains another dictionary with team names as
              keys and lists of sorted integer event times as values.
    """
    timeline = {
        'goals': defaultdict(list),
        'corners': defaultdict(list),
        'yellow_cards': defaultdict(list),
        'red_cards': defaultdict(list)
    }

    key_map = {
        'goal': 'goals',
        'corner': 'corners',
        'yellow_card': 'yellow_cards',
        'red_card': 'red_cards'
    }

    for event in events:
        event_type = event.get('event_type')
        team = event.get('team')
        time_str = event.get('time')

        timeline_key = key_map.get(event_type)

        if timeline_key and team and time_str:
            # Handle stoppage time like '45+2' and convert to integer
            try:
                if '+' in time_str:
                    parts = time_str.split('+')
                    time_int = int(parts[0]) + int(parts[1])
                else:
                    time_int = int(time_str)
                timeline[timeline_key][team].append(time_int)
            except (ValueError, IndexError):
                # Skip if time format is unexpected
                continue

    # Sort the times for each team within each event type
    for action_type in timeline.values():
        for team_times in action_type.values():
            team_times.sort()

    # Convert defaultdicts to regular dicts for a cleaner final output
    return {
        'goals': dict(timeline['goals']),
        'corners': dict(timeline['corners']),
        'yellow_cards': dict(timeline['yellow_cards']),
        'red_cards': dict(timeline['red_cards'])
    }


import re

def parse_live_match_data(response_text: str) -> dict:
    """
    Parses live match data from a raw text response, handling variable names,
    anonymous teams, and extracting the league link.

    This function uses robust regular expressions to find four specific charts
    ("On Target", "Off Target", "Dangerous Attacks", "Attacks"). It correctly
    locates the 'series' data regardless of the preceding Javascript variable name,
    assigns placeholder names to teams with empty name fields, and then corrects
    those placeholders with actual team names found elsewhere in the data.
    It also extracts the league's URL from the response.

    Args:
        response_text: The raw string data containing the match statistics.

    Returns:
        A dictionary structured as {chart_name: {team_name: [time1, time2, ...]},
        'league': '/league/id'}. Each chart value is a list of times in minutes
        for that team's actions, and the 'league' key holds the URL for the league.
    """
    # This pattern finds a title with one of the four key table names, then
    # non-greedily finds the *next* ".series = [...]" assignment that follows it.
    # Captures: 1. Table Name, 2. The full 'series' array string.
    chart_pattern = re.compile(
        r'"text":\s*"(On Target|Off Target|Dangerous Attacks|Attacks)[^"]*"' # Capture title
        r'.*?'                                     # Non-greedy match for anything in between
        r'\.series\s*=\s*(\[.*?\]);',               # Find the series assignment and capture the array
        re.DOTALL                                  # Allow '.' to match newlines
    )

    # This pattern finds each team's block inside a 'series' string.
    # It captures: 1. The team name (can be empty), 2. The team's 'data' array string.
    team_pattern = re.compile(
        r"'name':\s*'([^']*)'"   # Capture the content of 'name'
        r'.*?'                  # Non-greedy match
        r"'data':\s*(\[.*?\])",  # Capture the 'data' array
        re.DOTALL
    )

    # This pattern now extracts only the x coordinate (time) from a data array string.
    # It captures: 1. The 'x' value (time).
    points_pattern = re.compile(r'"x":\s*(\d+)')

    # This pattern finds and captures the league link.
    # It captures: 1. The URL starting with /league/.
    league_pattern = re.compile(r'href="[^"]*(/league/[^"]+)"')

    all_charts_data = {}

    # Find and add the league link to the dictionary
    league_match = league_pattern.search(response_text)
    all_charts_data['league'] = league_match.group(1) if league_match else None

    # Find all the main chart blocks in the text
    chart_matches = chart_pattern.findall(response_text)

    for table_name, series_string in chart_matches:
        all_charts_data[table_name] = {}
        team_matches = team_pattern.findall(series_string)
        anonymous_team_counter = 0

        for team_name, data_string in team_matches:
            final_team_name = team_name
            if not final_team_name:
                final_team_name = f"Team {anonymous_team_counter}"

            # Extract all time points for the current team
            points = points_pattern.findall(data_string)
            # The result is now a simple list of integers (times)
            timeseries = [int(x) for x in points]

            all_charts_data[table_name][final_team_name] = timeseries

            if not team_name:
                anonymous_team_counter += 1

    # --- Name Correction Step ---
    # First, find all non-placeholder names across all parsed data
    canonical_names = {0: "", 1: ""}
    for chart in all_charts_data.values():
        if isinstance(chart, dict): # Ensure we are iterating over a chart dictionary
            iiii=0
            for team_name in chart.keys():
                if not team_name.startswith("Team "):
                    canonical_names[iiii] = team_name
                iiii+=1

    # If no real names were found, no correction is possible.
    has_real_names = any(name for name in canonical_names.values())

    if has_real_names:
        for chart_name, chart_content in all_charts_data.items():
             if isinstance(chart_content, dict): # Ensure we are working with a chart dictionary
                if len(canonical_names) == 2:
                    if "Team 0" in chart_content and canonical_names.get(0):
                        chart_content[canonical_names[0]] = chart_content.pop("Team 0")
                    if "Team 1" in chart_content and canonical_names.get(1):
                        chart_content[canonical_names[1]] = chart_content.pop("Team 1")

    if "Attacks" in all_charts_data:
      del all_charts_data["Attacks"]
      
    return all_charts_data


import re
import pandas as pd
from io import StringIO
from datetime import datetime

# The parse_match_data function remains the same as it correctly parses the data.
def parse_match_data(snippet):
    # Regex to find all variable definitions and race array assignments
    # It captures blocks of variable assignments followed by a race assignment
    regex = re.compile(
        r'((?:tmp_host_name\s*=\s*".*?";\s*\n*\s*tmp_guest_name\s*=\s*".*?";\s*\n*\s*tmp_league_name\s*=\s*".*?";\s*\n*\s*)?race\[\d+\]\s*=\s*\[.*?\];)',
        re.DOTALL
    )
    
    # Find all matches in the snippet
    matches = regex.findall(snippet)
    
    all_races = []
    
    # Keep track of the most recent team names
    current_vars = {}

    for block in matches:
        # Extract tmp variables from the current block
        host_match = re.search(r'tmp_host_name\s*=\s*"(.*?)";', block)
        guest_match = re.search(r'tmp_guest_name\s*=\s*"(.*?)";', block)
        league_match = re.search(r'tmp_league_name\s*=\s*"(.*?)";', block)

        if host_match:
            current_vars['tmp_host_name'] = host_match.group(1)
        if guest_match:
            current_vars['tmp_guest_name'] = guest_match.group(1)
        if league_match:
            current_vars['tmp_league_name'] = league_match.group(1)

        # Extract the array part of the race assignment
        race_match = re.search(r'race\[\d+\]\s*=\s*(\[.*?\]);', block)
        if race_match:
            # Get the raw array string
            race_str = race_match.group(1)
            
            # Replace variable names with their actual values
            # Using replacement with quotes to make it a valid string literal
            for var, val in current_vars.items():
                race_str = race_str.replace(var, f"'{val}'")
            
            # Use a robust method to convert string to list
            try:
                # The string looks like a list, so we can use eval or literal_eval
                # We build a string that can be read by pandas read_csv
                # It's safer than eval
                s = race_str.replace('[', '').replace(']', '').replace("'", "")
                reader = pd.read_csv(StringIO(s), header=None)
                all_races.append(reader.iloc[0].tolist())
            except Exception as e:
                print(f"Could not parse row: {race_str}. Error: {e}")

    return all_races

def create_filtered_history_dataframes(races, home_team_focus, away_team_focus):
    """
    Processes a list of match data and converts it into two pandas DataFrames:
    - One for the home history of the home_team_focus.
    - One for the away history of the away_team_focus.

    Args:
        races: A list of lists, where each inner list contains data for one match.
        home_team_focus (str): The name of the team to generate home history for.
        away_team_focus (str): The name of the team to generate away history for.

    Returns:
        A tuple containing two pandas DataFrames: (home_history_df, away_history_df).
    """
    home_team_history = []
    away_team_history = []

    for race in races:
        # Clean up string data by stripping whitespace, if any
        race = [item.strip() if isinstance(item, str) else item for item in race]

        home_team = race[5]
        away_team = race[8]
        
        # Safely get scores, handling potential errors
        try:
            home_score = int(race[13])
            away_score = int(race[14])
        except (ValueError, IndexError):
            home_score, away_score = '-', '-'

        try:
            time_obj = datetime.strptime(race[2], '%y/%m/%d %H:%M')
        except (ValueError, IndexError):
            time_obj = None  # Or a default value like '-'

        # Common data structure for a match
        game_data = {
            'Match': race[12],
            'Time': time_obj,
            'Home': home_team,
            'Away': away_team,
            'Home Score': home_score,
            'Away Score': away_score,
            'Half Home Score': race[15],
            'Half Away Score': race[16],
            'H': race[9],
            'G': race[10],
            'C': race[11],
            'Half Home Corner': race[19],
            'Half Away Corner': race[20],
            'Full Home Corner': race[17],
            'Full Away Corner': race[18],
        }

        # Check if the current match is a home game for the team of interest
        if home_team == home_team_focus:
            entry = game_data.copy()
            if isinstance(home_score, int) and isinstance(away_score, int):
                if home_score > away_score:
                    entry['1X2'] = 'Win'
                elif home_score < away_score:
                    entry['1X2'] = 'Lose'
                else:
                    entry['1X2'] = 'Draw'
            else:
                entry['1X2'] = '-'
            home_team_history.append(entry)

        # Check if the current match is an away game for the team of interest
        if away_team == away_team_focus:
            entry = game_data.copy()
            if isinstance(home_score, int) and isinstance(away_score, int):
                if away_score > home_score:
                    entry['1X2'] = 'Win'
                elif away_score < home_score:
                    entry['1X2'] = 'Lose'
                else:
                    entry['1X2'] = 'Draw'
            else:
                entry['1X2'] = '-'
            away_team_history.append(entry)
            
    # Create DataFrames from the filtered lists
    home_df = pd.DataFrame(home_team_history)
    away_df = pd.DataFrame(away_team_history)
    
    # Reorder columns to match the desired output image
    column_order = [
        'Match', 'Time', 'Home', 'Away', 'Home Score', 'Away Score', 
        'Half Home Score', 'Half Away Score', 'H', 'G', 'C', '1X2', 
        'Half Home Corner', 'Half Away Corner', 'Full Home Corner', 'Full Away Corner'
    ]
    
    # Handle cases where a team might not have any home or away games in the data
    if not home_df.empty:
        # A simplified reordering just to place '1X2' correctly if not all columns exist
        cols = list(home_df.columns)
        cols.pop(cols.index('1X2'))
        home_df = home_df[['Match', 'Time', 'Home', 'Away', '1X2'] + cols[4:]]


    if not away_df.empty:
        cols = list(away_df.columns)
        cols.pop(cols.index('1X2'))
        away_df = away_df[['Match', 'Time', 'Home', 'Away', '1X2'] + cols[4:]]

    return home_df, away_df












def _get_average_stats(df, team_name):
    """
    Intelligently calculates average stats for a team from its match history.
    FIXED: Now correctly handles the data type to prevent AttributeError.
    """
    if df.empty or team_name is None:
        return {'avg_goals_for': 1.35, 'avg_goals_against': 1.35,
                'avg_corners_for': 5.25, 'avg_corners_against': 5.25}

    goals_for, goals_against = [], []
    corners_for, corners_against = [], []

    required_cols = ['Home', 'Away', 'Home Score', 'Away Score', 'Full Home Corner', 'Full Away Corner']
    if not all(col in df.columns for col in required_cols):
        return {'avg_goals_for': 1.35, 'avg_goals_against': 1.35,
                'avg_corners_for': 5.25, 'avg_corners_against': 5.25}

    for _, row in df.iterrows():
        if row['Home'] == team_name:
            goals_for.append(row['Home Score'])
            goals_against.append(row['Away Score'])
            corners_for.append(row['Full Home Corner'])
            corners_against.append(row['Full Away Corner'])
        elif row['Away'] == team_name:
            goals_for.append(row['Away Score'])
            goals_against.append(row['Home Score'])
            corners_for.append(row['Full Away Corner'])
            corners_against.append(row['Full Home Corner'])

    def _calculate_mean(data, default_value):
        """
        Helper to calculate mean safely.
        FIX: Explicitly converts list to a Pandas Series to ensure .dropna() method is available.
        """
        s = pd.Series(data) # Convert list to a Pandas Series
        numeric_series = pd.to_numeric(s, errors='coerce').dropna()
        return numeric_series.mean() if not numeric_series.empty else default_value

    return {
        'avg_goals_for': _calculate_mean(goals_for, 1.35),
        'avg_goals_against': _calculate_mean(goals_against, 1.35),
        'avg_corners_for': _calculate_mean(corners_for, 5.25),
        'avg_corners_against': _calculate_mean(corners_against, 5.25),
    }

def get_main_team_name_from_df(df):
    """
    (UNCHANGED) Robustly gets a team's name from its historical dataframe by finding the most frequent name.
    """
    if df.empty:
        return None
    try:
        all_names = pd.concat([df['Home'], df['Away']]).dropna()
        string_only_names = all_names[all_names.apply(lambda x: isinstance(x, str))]
        if string_only_names.empty:
            return None
        return string_only_names.mode()[0]
    except (KeyError, IndexError):
        return None

import pandas as pd

def create_features(timeline, current_minute, home_df, away_df, home_team_name, away_team_name, continent, gender, youth):
    """
    Creates a feature vector for a single minute, including categorical data.
    This version includes a robust method for fetching team stats to handle naming inconsistencies.
    """

    def _get_team_stats_robustly(stats_dict, team_name):
        """
        Fetches a team's stat list from a dictionary, even with partial name matches.
        """
        # First, try for an exact match.
        if team_name in stats_dict:
            return stats_dict[team_name]
        
        # If no exact match, find a key that is a substring of the full name, or vice versa.
        # This handles cases like 'Henan' vs. 'Henan Songshan Longmen'.
        for key, value in stats_dict.items():
            if team_name in key or key in team_name:
                return value
        
        # If no match is found, return an empty list.
        return []

    # Get historical average stats for both teams
    home_stats = _get_average_stats(home_df, home_team_name)
    away_stats = _get_average_stats(away_df, away_team_name)

    # Calculate pre-match expected goals and corners
    pre_match_exp_home_goals = (home_stats['avg_goals_for'] + away_stats['avg_goals_against']) / 2
    pre_match_exp_away_goals = (away_stats['avg_goals_for'] + home_stats['avg_goals_against']) / 2
    pre_match_exp_home_corners = (home_stats['avg_corners_for'] + away_stats['avg_corners_against']) / 2
    pre_match_exp_away_corners = (away_stats['avg_corners_for'] + home_stats['avg_corners_against']) / 2

    # --- FIX: Use the robust function to get in-match stats ---
    home_goals_list = _get_team_stats_robustly(timeline.get('goals', {}), home_team_name)
    away_goals_list = _get_team_stats_robustly(timeline.get('goals', {}), away_team_name)
    home_corners_list = _get_team_stats_robustly(timeline.get('corners', {}), home_team_name)
    away_corners_list = _get_team_stats_robustly(timeline.get('corners', {}), away_team_name)
    home_sot_list = _get_team_stats_robustly(timeline.get('On Target', {}), home_team_name)
    away_sot_list = _get_team_stats_robustly(timeline.get('On Target', {}), away_team_name)
    
    home_goals_so_far = len([g for g in home_goals_list if g <= current_minute])
    away_goals_so_far = len([g for g in away_goals_list if g <= current_minute])
    goal_diff = home_goals_so_far - away_goals_so_far
    home_corners_so_far = len([c for c in home_corners_list if c <= current_minute])
    away_corners_so_far = len([c for c in away_corners_list if c <= current_minute])
    home_sot = len([s for s in home_sot_list if s <= current_minute])
    away_sot = len([s for s in away_sot_list if s <= current_minute])

    # Get stats for the last 15 minutes to measure recent pressure
    last_15_min_start = max(0, current_minute - 15)
    home_sot_last_15 = len([s for s in home_sot_list if last_15_min_start < s <= current_minute])
    away_sot_last_15 = len([s for s in away_sot_list if last_15_min_start < s <= current_minute])

    # Compile all numerical features into a dictionary
    features = {
        'current_minute': current_minute, 'goal_diff': goal_diff,
        'home_goals_so_far': home_goals_so_far, 'away_goals_so_far': away_goals_so_far,
        'home_corners_so_far': home_corners_so_far, 'away_corners_so_far': away_corners_so_far,
        'home_sot': home_sot, 'away_sot': away_sot,
        'home_sot_last_15': home_sot_last_15, 'away_sot_last_15': away_sot_last_15,
        'pre_match_exp_home_goals': pre_match_exp_home_goals, 'pre_match_exp_away_goals': pre_match_exp_away_goals,
        'pre_match_exp_home_corners': pre_match_exp_home_corners, 'pre_match_exp_away_corners': pre_match_exp_away_corners,
    }

    # Process the newly added categorical features
    categorical_data = pd.DataFrame({'continent': [continent], 'gender': [gender], 'youth': [youth]})
    possible_categories = {
        'continent': ['asia', 'america', 'africa', 'europe', 'oceania'],
        'gender': ['men', 'women'],
        'youth': ['yes', 'no']
    }
    
    # One-hot encode the categorical features to be model-ready
    for col, categories in possible_categories.items():
        categorical_data[col] = pd.Categorical(categorical_data[col], categories=categories)
        
    encoded_features = pd.get_dummies(categorical_data, prefix=categorical_data.columns)
    
    # Add the encoded features to the main features dictionary
    features.update(encoded_features.to_dict(orient='records')[0])
    
    return features



def identify_home_away_teams(past_matches_for_id):
    """
    Identifies the home and away team names from their respective match history DataFrames.

    The function assumes that index 0 of the input list corresponds to the home team's
    historical data and index 1 corresponds to the away team's.

    Args:
        past_matches_for_id (list or tuple): A list containing two pandas DataFrames,
                                             the first for the home team's history,
                                             the second for the away team's history.

    Returns:
        tuple: A tuple containing the identified (home_team_name, away_team_name).
               Returns (None, None) if identification fails.
    """
    # Validate that the input is a list/tuple with two elements
    if not isinstance(past_matches_for_id, (list, tuple)) or len(past_matches_for_id) != 2:
        return None, None

    # Unpack the dataframes for home (index 0) and away (index 1)
    home_df, away_df = past_matches_for_id

    # Use the existing robust function to get the most common name from each dataframe
    home_team_name = get_main_team_name_from_df(home_df)
    away_team_name = get_main_team_name_from_df(away_df)

    return home_team_name, away_team_name

def predict_final_outcomes(
    match_id,
    current_minute,
    timeline_dict,
    past_matches_dict,
    goal_model,
    corner_model,
    goal_error_model,
    corner_error_model
):
    """
    UPDATED: Generates predictions for final goals and corners, including confidence intervals
    and team names, and returns them in a dictionary keyed by the match ID.

    Args:
        match_id (int or str): The unique identifier for the match.
        current_minute (int): The current minute of the match to predict from.
        timeline_dict (dict): A dictionary containing timeline data for all matches.
        past_matches_dict (dict): A dictionary containing historical match data for teams.
        goal_model: Trained XGBoost model for predicting remaining goals.
        corner_model: Trained XGBoost model for predicting remaining corners.
        goal_error_model: Trained XGBoost model for predicting the error of goal predictions.
        corner_error_model: Trained XGBoost model for predicting the error of corner predictions.

    Returns:
        dict: A dictionary where the key is the match_id and the value is another
              dictionary containing team names, predictions, errors, and confidence intervals.
              Returns an error dictionary if prediction fails.
    """
    # --- 1. Data Retrieval and Validation ---
    if match_id not in timeline_dict or match_id not in past_matches_dict:
        return {match_id: {"error": "Match ID not found in the provided data."}}

    timeline = timeline_dict[match_id]
    past_matches_for_id = past_matches_dict[match_id]

    # --- NEW: Identify team names using the helper function ---
    home_team_name, away_team_name = identify_home_away_teams(past_matches_for_id)

    if not home_team_name or not away_team_name:
        return {match_id: {"error": "Could not determine team names."}}

    home_df, away_df = past_matches_for_id

    continent = timeline.get('continent')
    gender = timeline.get('gender')
    youth = timeline.get('youth')

    if not all([continent, gender, youth]):
        return {match_id: {"error": "Missing categorical data (continent, gender, or youth)."}}

    # --- 2. Feature Creation ---
    try:
        features = create_features(
            timeline, current_minute, home_df, away_df,
            home_team_name, away_team_name, continent, gender, youth
        )
        features_df = pd.DataFrame([features])
        model_feature_names = goal_model.get_booster().feature_names
        features_df = features_df.reindex(columns=model_feature_names, fill_value=0)
    except Exception as e:
        return {match_id: {"error": f"An error occurred during feature creation: {e}"}}

    # --- 3. Prediction of Outcomes and Errors ---
    predicted_remaining_goals = goal_model.predict(features_df)[0]
    predicted_remaining_corners = corner_model.predict(features_df)[0]
    predicted_goal_error = max(0, goal_error_model.predict(features_df)[0])
    predicted_corner_error = max(0, corner_error_model.predict(features_df)[0])

    # --- 4. Final Outcome and Confidence Interval Calculation ---
    current_total_goals = features['home_goals_so_far'] + features['away_goals_so_far']
    current_total_corners = features['home_corners_so_far'] + features['away_corners_so_far']
    predicted_final_goals = current_total_goals + predicted_remaining_goals
    predicted_final_corners = current_total_corners + predicted_remaining_corners

    final_goal_confidence_interval = [
        max(current_total_goals, predicted_final_goals - predicted_goal_error),
        predicted_final_goals + predicted_goal_error
    ]
    final_corner_confidence_interval = [
        max(current_total_corners, predicted_final_corners - predicted_corner_error),
        predicted_final_corners + predicted_corner_error
    ]

    # --- 5. Structure the Output (with team names included) ---
    result = {
        match_id: {
            "home_team": home_team_name,
            "away_team": away_team_name,
            # "goals": f"{features['home_goals_so_far']} - {features['away_goals_so_far']}",
            # "corners": f"{features['home_corners_so_far']} - {features['away_corners_so_far']}",
            "home_goals": features['home_goals_so_far'],
            "away_goals": features['away_goals_so_far'],
            "home_corners": features['home_corners_so_far'],
            "away_corners": features['away_corners_so_far'],
            "current_minute": current_minute,
            "predicted_final_goals": float(predicted_final_goals),
            "predicted_final_corners": float(predicted_final_corners),
            "goal_prediction_error": float(predicted_goal_error),
            "corner_prediction_error": float(predicted_corner_error),
            "final_goal_confidence_interval": [float(val) for val in final_goal_confidence_interval],
            "final_corner_confidence_interval": [float(val) for val in final_corner_confidence_interval]
        }
    }
    return result














# Assume your models are in the same directory and loaded here
with open("goal_model.pkl", "rb") as f:
    goal_model = pickle.load(f)
with open("corner_model.pkl", "rb") as f:
    corner_model = pickle.load(f)
with open("goal_error_model.pkl", "rb") as f:
    goal_error_model = pickle.load(f)
with open("corner_error_model.pkl", "rb") as f:
    corner_error_model = pickle.load(f)

# --- Global Variables ---
session_id = '9vb3co7v023n3df35d1cn642i3'
csrf_token = "68807998ce4b9"

response_initial_scan = {}
responses = {}
timestamps = {}
timeline = {}
past_matches = {}

global_do_loop = False
global_id_info_loop = False
do_parse = False
abort = True

t_ids_thread = None
t_info_thread = None
t_parse_thread = None

t_timer_execute = None


to_get = [
    "match_live",
    "match_prediction",
    "match_statistics",
    "current_time",
    "parsed"
]

# --- Placeholder for your functions ---
# It's assumed that the following functions are defined here as you provided:
response_initial_scan = {}
def get_main():
    global response_initial_scan
    url = 'https://vip.scoremer.com/ajax/score/data'

    # The headers from the -H flags
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36',
    }

    cookies = {
        'ds_session': session_id,
    }

    # The payload from the --data-raw flag
    data = {
        'csrf_token': csrf_token,
    }

    try:
        # Make the POST request with all the components
        response_initial_scan = requests.post(
            url,
            headers=headers,
            cookies=cookies,
            data=data
        )

        # Raise an exception for bad status codes (4xx or 5xx)
        response_initial_scan.raise_for_status()

        # Print the status code and the response_initial_scan content
        print(f"Status Code: {response_initial_scan.status_code}")
        print("Response JSON:")
        response_initial_scan = {r["id"]: r for r in response_initial_scan.json()["rs"] if "ss" in r and r["ss"] == "S" and r["status"].isnumeric()}

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON. Raw response_initial_scan text:")
        print(response_initial_scan.text)

import time
import threading

global_do_loop = True
def call_get_main(timee):
    while global_do_loop:
        get_main()
        if not global_do_loop:
            break
        time.sleep(timee)
    print("call_get_main stopped")
def get_id_info(id, responses):
    print("do get id info", id)

    for tg in to_get:
        if id not in responses:
            responses[id] = {}
        if tg == to_get[1]:
            responses[id][tg] = ""
            continue
        if tg == to_get[3]:
            responses[id][tg] = time.time()
            continue
        if tg == to_get[4]:
            responses[id][tg] = False
            continue
        url = f'https://www.scoremer.com/{tg}/{id}'

        headers = {
            'referer': f'https://www.scoremer.com/{tg}/{id}',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
        }

        cookies = {
            # 'uid': 'R-366339-97a2d9b4068723a7d562da',
            'ds_session': session_id,
        }

        try:
            response = requests.get(url, headers=headers, cookies=cookies)
            response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)


            responses[id][tg] = response.text
            timestamps[id] = time.time()

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}, id: {id}, m: {tg}")

global_id_info_loop = True
def call_get_id_info(timee):
    while global_id_info_loop:
        global abort
        match_ids = [r for r in response_initial_scan]
        iter = 0
        for id in match_ids:
            if abort:
                print("aborted")
                break
            if id in responses:
                if id in timestamps:
                    elapsed_time = time.time() - timestamps[id]
                    if elapsed_time < 150:
                        print("skipped1", id)
                        continue
            
            get_id_info(id, responses)
            print("finished", iter, id)
            iter+=1
        # finished=True
        time.sleep(timee)
import re



def find_teams(long_string):
  """
  This function takes a long string and returns the two team names playing.
  """
  match = re.search(r"Historical Statistics of (.*?) vs (.*?) on", long_string)
  return match.group(1), match.group(2)

with open("leagues_info.pkl", "rb") as f:
    LEAGUES_INFO = pickle.load(f)

# for id, resps in responses.items():
def parse_id_info(id, responses, response_initial_scan, timeline, past_matches):
    for m, resp in responses[id].items():
        if m == to_get[0]:
            ret3 = parse_match_events_from_html(resp)
            ret4 = create_timeline_from_parsed_events(ret3)
            ret2 = parse_live_match_data(resp)
            if "league" not in ret2:
                print("LEAGUE not in", id)
                return [], []
            else:
                ret2["continent"] = LEAGUES_INFO[ret2["league"]]["continent"]
                ret2["gender"] = LEAGUES_INFO[ret2["league"]]["gender"]
                ret2["youth"] = LEAGUES_INFO[ret2["league"]]["youth"]
                del ret2["league"]
            # ret2 = filter_data_by_time(ret1)
            print("finished1", id, len(ret2), len(ret4))
            if len(ret2) <= 2:
                print("ret2 IS 0!!!!!!!!!!!!!!!!!!!!!!!")
                return [], []
            timeline[id] = ret2 | ret4
            
        if m == to_get[2]:
            parsed_races = parse_match_data(resp)

            if id in response_initial_scan:
                team1 = response_initial_scan[id]["host"]["n"]
                team2 = response_initial_scan[id]["guest"]["n"]
            else:
                team1, team2 = find_teams(resp)

            home_df, away_df = create_filtered_history_dataframes(
                parsed_races, 
                home_team_focus=team1, 
                away_team_focus=team2
            )

            past_matches[id] = [home_df, away_df]

            print("finished2", id, len(home_df), len(away_df))
    return ret2, ret4

do_parse = True
idd = []
def threaded_parse(timee):
    while do_parse:
        for id in response_initial_scan:
            idd.append(id)
            try:
                parse_id_info(id, responses, response_initial_scan, timeline, past_matches)
            except Exception as e:
                print("failed to parse", e, id)
                continue
        time.sleep(timee)
    print("threaded_parse aborted")



def get_good_time(str_time):
    if str_time.isnumeric():
        return int(str_time)
    spl = str_time.split("+")
    if len(spl) == 2 and spl[0].isnumeric() and spl[2].isnumeric():
        return int(spl[0])+int(spl[1])
    return None

def timer_execute(function, timee):
    time.sleep(timee)
    function()

def start_threads():
    """Starts all background threads."""
    global global_do_loop, global_id_info_loop, do_parse, abort
    global t_ids_thread, t_info_thread, t_parse_thread, t_timer_execute

    global_do_loop = True
    global_id_info_loop = True
    do_parse = True
    abort = False

    if t_ids_thread is None or not t_ids_thread.is_alive():
        t_ids_thread = threading.Thread(target=call_get_main, args=(60,))
        t_ids_thread.start()

    if t_info_thread is None or not t_info_thread.is_alive():
        t_info_thread = threading.Thread(target=call_get_id_info, args=(30,))
        t_info_thread.start()

    if t_parse_thread is None or not t_parse_thread.is_alive():
        t_parse_thread = threading.Thread(target=threaded_parse, args=(30,))
        t_parse_thread.start()
    
    if t_timer_execute is None or not t_timer_execute.is_alive():
        t_timer_execute = threading.Thread(target=timer_execute, args=(stop_threads, 30))
        t_timer_execute.start()
    print("Threads started")

def stop_threads():
    """Stops all background threads."""
    global global_do_loop, global_id_info_loop, do_parse, abort
    global_do_loop = False
    global_id_info_loop = False
    do_parse = False
    abort = True
    print("Threads stopped")




def run_predictions():
    """
    Runs predictions on all matches in response_initial_scan and returns the results.
    """
    predictions = []
    for iddd in set(response_initial_scan):
        current_minute = get_good_time(response_initial_scan[iddd]["status"])
        if current_minute is None:
            continue

        prediction = predict_final_outcomes(
            match_id=iddd,
            current_minute=current_minute,
            timeline_dict=timeline,
            past_matches_dict=past_matches,
            goal_model=goal_model,
            corner_model=corner_model,
            goal_error_model=goal_error_model,
            corner_error_model=corner_error_model
        )
        if iddd in prediction and "error" not in prediction[iddd]:
            predictions.append(prediction[iddd])
    return predictions