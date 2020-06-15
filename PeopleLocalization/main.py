import pandas as pd


# this module takes as input a name of a painting and find in which room the painting is
def get_room(id):
    with open('data.csv') as f:
        df = pd.read_csv(f)
        room = df.loc[df['Title'] == id]['Room']
    return room
