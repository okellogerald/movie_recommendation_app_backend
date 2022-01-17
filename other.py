
from random import randint
from app import database

def generate_fake_rating():
    for i in range(30):
        index = i + 1000
        ratings = []
        for rate in range(52):
            ratings.append(randint(0,4))
        database.collection('Ratings').document(str(index)).set({'ratings': ratings})
    print('ALL DONE, BONNE REVOIUR')



generate_fake_rating()