import pandas as pd
import cv2

# positions taken from the resized map image
# ToDo: decide the shape of map.png that will be displayed to the user and store in this dict all rooms' centers
map_grid = {1: (281, 145), 2: (281, 187), 3: (251, 187), 4: (221, 187), 5: (182, 187)}


# takes as input a name of a painting and find in which room the painting is
def get_room(id):
    with open('data.csv') as f:
        df = pd.read_csv(f)
        room = df.loc[df['Title'] == id].iloc[0]['Room']
    return room


# print on map.png a red circle on the room passed as input
# useful for user visualization (this function will be used to display on GUI where a person is)
def print_on_map(room):
    map = cv2.imread('images/map.png')
    height, width, depth = map.shape
    imgScale = 300 / width
    newX, newY = map.shape[1] * imgScale, map.shape[0] * imgScale
    new_map = cv2.resize(map, (int(newX), int(newY)))
    center = map_grid[room]
    cv2.circle(new_map, center, 10, (0, 0, 255), -1)
    cv2.imshow('map', new_map)
    cv2.waitKey()


if __name__ == "__main__":
    room = get_room("Sant'Antonio da Padova")
    print_on_map(room)
