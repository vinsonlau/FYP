import firebase_admin
from firebase_admin import credentials
import numpy
import urllib
import pyrebase
from datetime import datetime, timedelta
from threading import Timer
from django import template

class AVFirebase():
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

    def __init__(self):
        self.firebase = pyrebase.initialize_app(self.FIREBASE_CONFIG)
        self.db = self.firebase.database()
        self.auth = self.firebase.auth()

    def sign_in(self, email, password):
        try:
            self.auth.sign_in_with_email_and_password(email, password)
            return True
        except:
            return False

    def upload_file(self, filename, cloudfilename):
        storage.child(cloudfilename).put(filename)

    def download_file(self, cloudfilename, filename, path=""):
        storage.child(cloudfilename).download(path, filename)

    def read_file(self, cloudfilename):
        url = storage.child(cloudfilename).get_url(None)
        file_stream = urllib.request.urlopen(url).read()
        return file_stream

    def write_database(self, data):
        num_children = self.get_children_count()+1

        """
        #assume aloe vera count ranges from 0-99 only
        #same method with below, but this requires more lines
        if(num < 10):
            av_id = "av0" + str(num)
        else:
            av_id = "av" + str(num)
        print(av_id)
        """
        av_id = "av{:02}".format(num_children)

        self.db.child('aloevera').child(av_id).set(data)

    def update_database(self, data, node):
        #self.db.child(node).update(data)
        self.db.child('aloevera').child(node).update(data)

    def get_aloe_vera(self, id):
        av = self.db.child('aloevera').child(id).get()
        return av.val()

    def update_aloe_vera(self, id, data):
        av = self.get_aloe_vera(id)
        av_dic = {
            'condition': av['condition'],
            'datetime': av['datetime'],
            'height': av['height'],
            'width': av['width']
        }

        num_histories = self.get_history_count(id) + 1
        hid = "h{:02}".format(num_histories)
        self.db.child('aloevera').child(id).update(data)
        self.db.child('aloevera').child(id).child('histories').child(hid).set(av_dic)

    def get_children_count(self):
        num = len(self.db.child('aloevera').get().val())
        return num

    def get_history_count(self,id):
        num_histories = len(self.db
                            .child('aloevera')
                            .child(id)
                            .get()
                            .val()['histories'])
        return num_histories

    def print_hello():
        print('hello')
"""
data = {
        'Condition': 'Mature',
        'Height':'360cm',
        'History':{
            'Condition': 'Mature',
            'Height':'600cm',
            }
}
"""

#data extracted from aloe vera at the current time
data = {
      "condition": "baby",
      "datetime": "29/10/2020 08:00:00",
      "height": 155,
      "width": 155
    }

firebase = AVFirebase()
#print(firebase.get_aloe_vera("av03"))
#firebase.write_database(data)
#firebase.update_database(data, 'av02')
#firebase.update_aloe_vera("av01", data)

test = firebase.get_aloe_vera("av01")
#print(firebase.get_aloe_vera("av01"))

for x in test:
    if(x == 'condition'):
        print (test.get('condition'))