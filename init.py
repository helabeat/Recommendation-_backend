from flask import Flask, request, jsonify
import json

import firebase_admin 
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore, initialize_app
from flaskext.mysql import MySQL

from controller.user_artist_recommendation import recommender
from controller.hybrid_dynamic_playlist_generation import Dynamic_Palylist_Generation
from model.connection import Connection

app = Flask(__name__)
app.config["DEBUG"] = True
conn = Connection()
conn.initiate_connection(app)

# cred = credentials.Certificate("utils\key.json")
# firebase_admin.initialize_app(cred, {'databaseURL': 'https://helabeat-user-data.firebaseio.com'})

# Initialize Firestore DB
cred = credentials.Certificate('utils\key.json')
default_app = initialize_app(cred)
db = firestore.client()


@app.route('/getHistory')
def hello_world():
    try:
        user_id = request.args.get('id')   
        history_id = ''
        todo_ref = db.collection('users/'+user_id+'/history')
        if history_id:
            print('users/'+user_id+'/history')
            todo = todo_ref.document(user_id).get()
            return jsonify(todo.to_dict()), 200
        else:
            print('else')
            all_todos = [doc.to_dict() for doc in todo_ref.stream()]
            return jsonify(all_todos), 200
    except Exception as e:
        return f"An Error Occured: {e}"


@app.route('/test')
def test():
    user_id = request.args.get('user_id')
    print(user_id)

    rs = recommender()
    todo_ref = rs.recommend_songs(int(user_id))
    data = conn.query(todo_ref)
    return jsonify(data), 200

@app.route('/test2')
def test2():
    data = conn.query()
    return jsonify(data), 200

@app.route('/test3')
def test3():
    dpg = Dynamic_Palylist_Generation()
    dpg.baselineMethod()
    # dpg.KNN_model(user_listen, songs)
    dpg.DPG_recommendation('Ru Sara') 
    # dpg.Fasttext_model()
    data = dpg.recommend_songs(user_id = 10296, new_song = 'Saragee Asille')
    print(data)

if __name__ == '__main__':
    app.run(debug=True)