import gradio as gr
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from catboost import CatBoostRegressor
import mlbstatsapi
import tensorflow as tf


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def predict(inning, game_id):

    if inning == "Seven":
        inning = 7
    elif inning == "Eight":
        inning = 8
    elif inning == "Five":
        inning = 5
    elif inning == "Six":
        inning = 6
    
    df = data_retrieve(inning, game_id)
    # print(df)

    df_home = df[df['Home/Away'] == "Home"]
    df_away = df[df['Home/Away'] == "Away"]
    
    if (len(df_home) < inning) or (len(df_away) < inning):
        return [None, None]

    # print(df_home)

    df_main = pd.read_csv("Score_prediction_dataset_25th_August_TS_3seas.csv")
    df_main = df_main.drop(columns=['Opp_LOB'])
    df_main = df_main[(df_main['Inning'] <= inning)]

    df_main = df_main[df_main['Team_Name'].isin(team_names)]
    df_main = df_main[df_main['Opposition_Team'].isin(team_names)]

    df_main = df_main[(df_main['Team_Name'] != 'American League All-Stars') & (df_main['Team_Name'] != 'National League All-Stars')]
    df_main = df_main.dropna()
    df_main = df_main.drop_duplicates()

    df_main = df_main.pivot(index=['Game_ID', 'Team_Name', 'Opposition_Team', 'Home/Away', 'Final_Score'], columns='Inning', values=['Hits', 'Opp_Hits', 'Errors', 'Runs', 'Opp_Runs', 'LOB'])
    df_main.columns = [f'{feature}_{inning}' for feature, inning in df_main.columns]
    df_main = df_main.reset_index()
    df_main = df_main.drop(columns=['Final_Score'])
    df_main = pd.get_dummies(df_main, columns=['Team_Name', 'Opposition_Team'])

    # df = pd.DataFrame([data], columns=["Team_Name", "Opposition_Team", "Inning", "Home/Away", "Hits", "Opp_Hits", "Errors", "Runs", "Opp_Runs", "LOB"])

    pivoted_df_home = df_home.pivot(index=['Game_ID', 'Team_Name', 'Opposition_Team', 'Home/Away', 'Final_Score'], columns='Inning', values=['Hits', 'Opp_Hits', 'Errors', 'Runs', 'Opp_Runs', 'LOB'])
    pivoted_df_home.columns = [f'{feature}_{inning}' for feature, inning in pivoted_df_home.columns]
    # print(pivoted_df_home)
    pivoted_df_home = pivoted_df_home.reset_index()

    pivoted_df_home = pd.get_dummies(pivoted_df_home, columns=['Team_Name', 'Opposition_Team'])
    pivoted_df_home = pivoted_df_home.reindex(columns=df_main.columns, fill_value=0)

    pivoted_df_away = df_away.pivot(index=['Game_ID', 'Team_Name', 'Opposition_Team', 'Home/Away', 'Final_Score'], columns='Inning', values=['Hits', 'Opp_Hits', 'Errors', 'Runs', 'Opp_Runs', 'LOB'])
    pivoted_df_away.columns = [f'{feature}_{inning}' for feature, inning in pivoted_df_away.columns]
    pivoted_df_away = pivoted_df_away.reset_index()

    pivoted_df_away = pd.get_dummies(pivoted_df_away, columns=['Team_Name', 'Opposition_Team'])
    pivoted_df_away = pivoted_df_away.reindex(columns=df_main.columns, fill_value=0)

    print(pivoted_df_home)
    home_away_status = {'Home': 0, 'Away': 1}
    pivoted_df_home['Home/Away'] = pivoted_df_home['Home/Away'].map(home_away_status)
    pivoted_df_away['Home/Away'] = pivoted_df_away['Home/Away'].map(home_away_status)

    pivoted_df_home = pivoted_df_home.astype(int)
    pivoted_df_away = pivoted_df_away.astype(int)

    pivoted_df_home = pivoted_df_home.drop(['Game_ID'], axis=1)
    pivoted_df_away = pivoted_df_away.drop(['Game_ID'], axis=1)

    # return

    # print(len(df.columns))
    if inning == 8:
        model = tf.keras.models.load_model('ANNR_ts_CLAS_inn8_exp8_model.keras')
    elif inning ==7:
        model = tf.keras.models.load_model('ANNR_ts_CLAS_inn7_exp11_model.keras')
    elif inning ==6:
        model = tf.keras.models.load_model('ANNR_ts_CLAS_inn6_exp7_model.keras')
    elif inning ==5:
        model = tf.keras.models.load_model('ANNR_ts_CLAS_inn5_exp4_model.keras')
    

    # with open('pca_model4.pkl', 'rb') as f:
    #     pca = pickle.load(f)

    # with open('label_encoder_teams_xgbr1_exp3.pkl', 'rb') as f:
    #     label_encoder = pickle.load(f)
    
    # print(pivoted_df_home)

    # df = pca.transform(df)
    # return
    winner_prob_1 = model.predict(pivoted_df_home)
    winner__prob_2 = model.predict(pivoted_df_away)


    if winner_prob_1 < winner__prob_2:
        winner = 'Home_Team'
    else:
        winner = 'Away_Team'
        # return score_1
    
    return winner

def data_retrieve(inning, game_id):
    mlb = mlbstatsapi.Mlb()
    df = pd.DataFrame(columns = ["Game_ID", "Team_Name", "Opposition_Team", "Inning", "Home/Away", "Hits", "Opp_Hits", "Errors", "Runs", "Opp_Runs", "LOB", "Opp_LOB", "Final_Score"])

    try:
        linescore = mlb.get_game_line_score(game_id, verify = False)
    except:
        gr.Info("Error retrieving data!!!")
    
    home_runs = 0
    away_runs = 0
    home_hits = 0
    away_hits = 0
    home_errors = 0
    away_errors = 0
    home_leftonbase = 0
    away_leftonbase = 0
    count = 0
    for i in range(inning):
        try:
            inning = linescore.innings[i].num
            home_team_name = mlb.get_game_box_score(game_id, verify = False).teams.home.team.name
            away_team_name = mlb.get_game_box_score(game_id, verify = False).teams.away.team.name
            
            home_runs += linescore.innings[i].home.runs
            away_runs += linescore.innings[i].away.runs
            home_hits += linescore.innings[i].home.hits
            away_hits += linescore.innings[i].away.hits
            home_errors += linescore.innings[i].home.errors
            away_errors += linescore.innings[i].away.errors
            home_leftonbase += linescore.innings[i].home.leftonbase
            away_leftonbase += linescore.innings[i].away.leftonbase

            home_score = linescore.teams.home.runs
            away_score = linescore.teams.away.runs
        except:
            gr.Info(f"Error retrieving inning {i+1} data!!!")
            continue

        home_dict = {"Game_ID": game_id, "Team_Name": home_team_name, "Opposition_Team": away_team_name, "Inning": inning, "Home/Away": 'Home', "Hits": home_hits, "Opp_Hits": away_hits, "Errors": home_errors, "Runs": home_runs, "Opp_Runs": away_runs, "LOB": home_leftonbase, "Opp_LOB": away_leftonbase, "Final_Score": home_score}
        away_dict = {"Game_ID": game_id, "Team_Name": away_team_name, "Opposition_Team": home_team_name, "Inning": inning, "Home/Away": 'Away', "Hits": away_hits, "Opp_Hits": home_hits, "Errors": away_errors, "Runs": away_runs, "Opp_Runs": home_runs, "LOB": away_leftonbase, "Opp_LOB": home_leftonbase, "Final_Score": away_score}

        #print(home_dict)

        home_df = pd.DataFrame([home_dict])
        away_df = pd.DataFrame([away_dict])
        df = pd.concat([df,home_df], ignore_index=True)
        df = pd.concat([df,away_df], ignore_index=True)
        count += 1

    if count != inning:
        gr.Info("All reuiqred innings are not available!!!")

    return df

team_names = ["Arizona Diamondbacks",
"Atlanta Braves",
"Baltimore Orioles",
"Boston Red Sox",
"Chicago Cubs",
"Chicago White Sox",
"Cincinnati Reds",
"Cleveland Guardians",
"Colorado Rockies",
"Detroit Tigers",
"Houston Astros",
"Kansas City Royals",
"Los Angeles Angels",
"Los Angeles Dodgers",
"Miami Marlins",
"Milwaukee Brewers",
"Minnesota Twins",
"New York Mets",
"New York Yankees",
"Oakland Athletics",
"Philadelphia Phillies",
"Pittsburgh Pirates",
"San Diego Padres",
"San Francisco Giants",
"Seattle Mariners",
"St. Louis Cardinals",
"Tampa Bay Rays",
"Texas Rangers",
"Toronto Blue Jays",
"Washington Nationals"]

with gr.Blocks() as demo:
    # gr.Image("../Documentation/Context Diagram.png", scale=2)
    # gr(title="Your Interface Title")
    gr.Markdown("""
                <center> 
                <span style='font-size: 50px; font-weight: Bold; font-family: "Graduate", serif'>
                MLB Score Predictor V2
                </span>
                </center>
                """)
    # gr.Markdown("""
    #             <center> 
    #             <span style='font-size: 30px; line-height: 0.1; font-weight: Bold; font-family: "Graduate", serif'>
    #             Admin Dashboard 
    #             </span>
    #             </center>
    #             """)
    with gr.Row():
        inning = gr.Radio(["Five", "Six", "Seven", "Eight"], label="Inning", scale=1)
        game_id = gr.Number(None, minimum=0, label="Game_ID", scale=1)

    
    # with gr.Row():
    #     # with gr.Column():
    #     #     # venue = gr.Dropdown(choices = ["Home", "Away"], value="Away", max_choices = 1, label="Home/Away Status", scale=1)  
    #     #     inning = gr.Number(None, label="Inning", minimum = 1, maximum = 8, scale=1)

        
    #     with gr.Column():
    #         # opp_venue = gr.Dropdown(choices = ["Home", "Away"], value="Home", max_choices = 1, label="Opposition Home/Away Status", scale=1)  
    #         game_id = gr.Number(None, minimum=0, label="Game_ID", scale=1)

    
    # with gr.Row():
    #     with gr.Column():
    #         team = gr.Dropdown(choices = team_names, max_choices = 1, label="Team", scale=1)  
        
    #     with gr.Column():
    #         opp_team = gr.Dropdown(choices = team_names, max_choices = 1, label="Opposition Team", scale=1) 
    
    # with gr.Row():
    #     with gr.Column():
    #         hits = gr.Number(None, minimum=0, label="Hits - (H)", scale=1)
            
    #     with gr.Column():
    #         opp_hits = gr.Number(None, minimum=0, label="Opposition Hits - (H)", scale=1)
    
    # # summarize_btn = gr.Button(value="Summarize Text", size = 'sm')

    # with gr.Row():
    #     with gr.Column():
    #         errors = gr.Number(None, minimum=0, label="Errors - (E)", scale=2)
        
    #     with gr.Column():
    #         opp_errors = gr.Number(None, minimum=0, label="Opposition Errors - (E)", scale=2)

    #     # runs = gr.Number(None, minimum=0, label="Runs - (R)", scale=1)

    # with gr.Row():
    #     with gr.Column():
    #         lob = gr.Number(None, minimum=0, label="Left on Base - (LOB)", scale=1)

    #     with gr.Column():
    #         opp_lob = gr.Number(None, minimum=0, label="Opposition Left on Base - (LOB)", scale=1)

    # with gr.Row():
    #     with gr.Column():
    #         runs = gr.Number(None, minimum=0, label="Runs - (R)", scale=1)

    #     with gr.Column():
    #         opp_runs = gr.Number(None, minimum=0, label="Opposition Runs - (R)", scale=1)

    with gr.Row():
        predict_btn = gr.Button(value="Predict", size = 'sm')

    with gr.Row():
        with gr.Column():
            Winning_Team = gr.Textbox(label="Predicted Winner", scale=1)

        # with gr.Column():
        #     final_score_home1 = gr.Textbox(label="Home Team Predicted Score", scale=1)

    # with gr.Row():
    #     with gr.Column():
    #         final_score_away2 = gr.Textbox(label="Predicted Score Model CATB", scale=1)

    #     with gr.Column():
    #         final_score_home2 = gr.Textbox(label="Opposition Predicted Score Model CATB", scale=1)

    # patent_doc.upload(document_to_text, inputs = [patent_doc, slider, select_model], outputs=summary_doc)
    predict_btn.click(predict, inputs=[inning, game_id], outputs=[Winning_Team])
    # predict_btn.click(predict, inputs=[inning, game_id], outputs=final_score_home1)

    # predict_btn.click(predict_2, inputs=[team, inning, venue, hits, errors, lob, runs, opp_team, opp_runs, opp_hits], outputs=final_score_away2)
    # predict_btn.click(predict_2, inputs=[opp_team, inning, opp_venue, opp_hits, opp_errors, opp_lob, opp_runs, team, runs, hits], outputs=final_score_home2)

demo.launch(inbrowser=True)