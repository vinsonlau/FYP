import firebase_admin
from firebase_admin import credentials
import numpy
import urllib
import pyrebase
from datetime import datetime, timedelta
from threading import Timer
from django import template
register = template.Library()

FIREBASE_CONFIG = {
       "apiKey": "AIzaSyD94vOwL4QElyC5rE1J_-3nQm1CjqveMOg",
       "authDomain": "final-year-project-def12.firebaseapp.com",
       "databaseURL": "https://final-year-project-def12.firebaseio.com",
       "projectId": "final-year-project-def12",
       "storageBucket": "final-year-project-def12.appspot.com",
       "messagingSenderId": "287247073978",
       "appId": "1:287247073978:web:971ebe396135fb30c67883",
       "measurementId": "G-VQ06E09C78"
   }

def firebase_initialization():
    firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
    return firebase

@register.simple_tag
def write_database(data):
    firebase = firebase_initialization()
    num_children = self.get_children_count()+1
    av_id = "av{:02}".format(num_children)
    firebase.database().child('aloevera').child(av_id).set(data)

@register.simple_tag
def update_database(data, node):
    firebase = firebase_initialization()
    firebase.database().child('aloevera').child(node).update(data)

@register.simple_tag
def get_aloe_vera(id):
    firebase = firebase_initialization()
    av = firebase.database().child('aloevera').child(id).get()
    return av.val()

@register.simple_tag
def get_aloe_vera_history(avid, hid):
    firebase = firebase_initialization()
    av = firebase.database().child('aloevera').child(avid).child('histories').child(hid).get()
    return av.val()

@register.simple_tag
def update_aloe_vera(id, data):
    firebase = firebase_initialization()
    av = self.get_aloe_vera(id)
    av_dic = {
        'condition': av['condition'],
        'datetime': av['datetime'],
        'height': av['height'],
        'width': av['width']
    }

    num_histories = self.get_history_count(id) + 1
    hid = "h{:02}".format(num_histories)
    firebase.database().child('aloevera').child(id).update(data)
    firebase.database().child('aloevera').child(id).child('histories').child(hid).set(av_dic)

@register.simple_tag
def get_children_count():
    firebase = firebase_initialization()
    num = len(firebase.database().child('aloevera').get().val())
    return num

@register.simple_tag
def get_history_count(id):
    firebase = firebase_initialization()
    num_histories = len(firebase.database()
                        .child('aloevera')
                        .child(id)
                        .get()
                        .val()['histories'])
    return num_histories

@register.simple_tag
def to_str(value):
    return str(value)
