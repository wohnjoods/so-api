"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from __future__ import print_function
from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask import jsonify
from datetime import datetime
import datetime
from time import strptime
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences 
from keras.models import load_model
import numpy as np
import keras.models
from keras import backend as K
import tensorflow as tf
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#db_connect = create_engine('sqlite:///chinook.db')
import pyodbc 
#IMPORT MODULES
from statistics import mean 

import sys
import json
import csv

from keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import os


#STUFF TO INITIATE BEFORE STARTING API
####################################################################################################
server = 'tcp:sportsoracle.database.windows.net' 
database = 'ncaab' 
username = 'woodsjo' 
password = 'joW*0102' 
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

#Initiate App
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
api = Api(app)

#SOME GLOBAL VARS TO BE USED BY API
global mean, std
scaler = StandardScaler()
lineScale = pd.read_sql('SELECT line FROM scaler', con=cnxn)
lineScale.dropna(how='all',inplace=True)
mean = lineScale.mean()
mean = mean.values[0]
std = lineScale.std(ddof=0)
std = std.values[0]

K.clear_session()
global ncaab_spread_model,ncaab_win_model , graph1, graph2

ncaab_spread_model = load_model('full_season_spread_weights_prod.h5')
graph1 = tf.get_default_graph() 

ncaab_win_model = load_model('full_season_win_weights_prod.h5')
graph2 = tf.get_default_graph()

#GLOBAL METHODS TO BE USED BY API
def convertYearToSeason(GameDate):
    Year = GameDate.year
    if(GameDate.month<=12 and GameDate.month>=11):
        Season=GameDate.year+1
    else:
        Season=GameDate.year
    return Season
def getDayNum(GameDate,Season):
    season = pd.read_sql('SELECT * FROM season', con=cnxn)
    season=season[['Season','DayZero']]
    ZeroDate = season[(season['Season']==Season)].DayZero
    ZeroDate = ZeroDate.values[0]
    ZeroDate = pd.to_datetime(ZeroDate)
    DayNum=GameDate-ZeroDate
    DayNum=DayNum.days
    return DayNum
#Used to get Season and dayNum from a date
def convertDate(date):
    GameDate = datetime.datetime.strptime(date, '%m-%d-%Y')
    Season=convertYearToSeason(GameDate)
    dayNum = getDayNum(GameDate,Season)
    return Season, dayNum

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/ncaab/boxscores/<date>')
def ncaab_boxscore_date(date):
    #Season, dayNum = convertDate(date)
    result = cursor.execute("""SELECT * FROM boxscores Where date = ?""",date) 
    items = [dict(zip([key[0] for key in cursor.description], row)) for row in result]
    return jsonify(items)
@app.route('/ncaab/schedule/<date>')
def ncaab_schedule_date(date):
    #convert date to Season and dayNum, i.e. 11-22-2018 = 2019, 17
    #Season, dayNum = convertDate(date)
    result = cursor.execute("""SELECT * FROM schedule Where date = ?""",date) 
    items = [dict(zip([key[0] for key in cursor.description], row)) for row in result]
    return jsonify(items)
@app.route('/ncaab/schedule/<date>/<HTeamID>/<RTeamID>')
def ncaab_schedule_instance(date, HTeamID, RTeamID):
    """Renders a sample page."""
    Season, dayNum = convertDate(date)
    result = cursor.execute("""SELECT * FROM schedule Where date = ? and HTeamID = ? and RTeamID = ?"""
                            ,date, HTeamID, RTeamID) 
    items = [dict(zip([key[0] for key in cursor.description], row)) for row in result]
    return jsonify(items)
@app.route('/ncaab/model/inputs/<date>/<ATeamID>/<BTeamID>')
def ncaab_model_input(date, ATeamID, BTeamID):
    while True:
            result = cursor.execute("""SELECT TOP 1 * FROM inputs Where date = ? and ATeamID = ? and BTeamID = ? """
                            ,date, ATeamID, BTeamID) 
            if result.rowcount == -1:
                break
            else:
                result = cursor.execute("""SELECT TOP 1 * FROM inputs Where date = ? and  BTeamID = ? and ATeamID = ?"""
                                    ,date, ATeamID, BTeamID) 
                if result.rowcount == -1:
                    break
                else:
                    date = datetime.datetime.strptime(date, '%m-%d-%Y')
                    date = date - datetime.timedelta(days=1)
                    date = datetime.datetime.strftime(date, '%m-%d-%Y')
    columns = [column[0] for column in cursor.description]
    inputs = list(result.fetchone())
    items = [dict(zip(columns, inputs))]
    return jsonify(items)
@app.route('/ncaab/spread/outputs/<date>/<ATeamID>/<BTeamID>/<line>')
def ncaab_spread_output(date, ATeamID, BTeamID,line):
    while True:
            result = cursor.execute("""SELECT TOP 1 * FROM inputs Where date = ? and ATeamID = ? and BTeamID = ? """
                            ,date, ATeamID, BTeamID) 
            if result.rowcount == -1:
                break
            else:
                result = cursor.execute("""SELECT TOP 1 * FROM inputs Where date = ? and  BTeamID = ? and ATeamID = ?"""
                                    ,date, ATeamID, BTeamID) 
                if result.rowcount == -1:
                    break
                else:
                    date = datetime.datetime.strptime(date, '%m-%d-%Y')
                    date = date - datetime.timedelta(days=1)
                    date = datetime.datetime.strftime(date, '%m-%d-%Y')
    end =[]
    inputs = list(result.fetchone())
    end = inputs[50:51] + inputs[53:54]
    inputs = inputs[:45]
    #SCALE LINE DATA
        
    line=float(line)
    line = (line - mean)/std
        
        
    inputs.append(line)
    inputFeature = np.asarray(inputs).reshape(1, 46)
    with graph1.as_default():
        raw_pred = ncaab_spread_model.predict(inputFeature)[0][0]
        
    if raw_pred>.5:
        pred = end[0]
    else:
        pred = end[1]
    end.append(str(raw_pred))
    end.append(pred)
    #end.append(line)
    dictEnd = {
        "ATeam" : end[0],
        "BTeam" : end[1],
        "raw_prediction" : end[2],
        "prediction" : end[3]
    }
    return jsonify(dictEnd)
@app.route('/ncaab/win/outputs/<date>/<ATeamID>/<BTeamID>')
def ncaab_win_output(date, ATeamID, BTeamID):
    while True:
            result = cursor.execute("""SELECT TOP 1 * FROM inputs Where date = ? and ATeamID = ? and BTeamID = ? """
                            ,date, ATeamID, BTeamID) 
            if result.rowcount == -1:
                break
            else:
                result = cursor.execute("""SELECT TOP 1 * FROM inputs Where date = ? and  BTeamID = ? and ATeamID = ?"""
                                    ,date, ATeamID, BTeamID) 
                if result.rowcount == -1:
                    break
                else:
                    date = datetime.datetime.strptime(date, '%m-%d-%Y')
                    date = date - datetime.timedelta(days=1)
                    date = datetime.datetime.strftime(date, '%m-%d-%Y')
    end =[]
    inputs = list(result.fetchone())
    end = inputs[50:51] + inputs[53:54]
    inputs = inputs[:45]
    #SCALE LINE DATA
        
        
    inputFeature = np.asarray(inputs).reshape(1, 45)
    with graph2.as_default():
        raw_pred = ncaab_win_model.predict(inputFeature)[0][0]
        
    if raw_pred>.5:
        pred = end[0]
    else:
        pred = end[1]
    end.append(str(raw_pred))
    end.append(pred)
    #end.append(line)
    dictEnd = {
        "ATeam" : end[0],
        "BTeam" : end[1],
        "raw_prediction" : end[2],
        "prediction" : end[3]
    }
    return jsonify(dictEnd)

if __name__ == '__main__':
    app.run()
