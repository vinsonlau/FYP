from firebase_admin import credentials
from firebase_admin import firestore
import firebase_admin


cred = credentials.Certificate('config_key.json')

firebase_admin.initialize_app(cred)

db = firestore.client()

aloevera = db.collection('aloevera').stream()

for i in av01:
    print("{0} => {1}".foramt(i.id, i.to_dict()))

dic = doc_ref.to_dict()

print(av01['height'])