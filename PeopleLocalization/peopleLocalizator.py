import pandas as pd
import cv2

# positions taken from the resized map image
map_grid = {1: (375, 216), 2: (375, 275), 3: (335, 275), 4: (295, 275), 5: (255, 275), 6: (215, 275),
            7: (176, 275), 8: (147, 275), 9: (117, 275), 10: (89, 275), 11: (61, 275), 12: (21, 275),
            13: (21, 215), 14: (21, 150), 15: (21, 90), 16: (30, 27), 17: (70, 27), 18: (102, 27),
            19: (83, 109), 20: (98, 215), 21: (218, 215), 22: (313, 215)}


# takes as input a name of a painting and find in which room the painting is
def get_room(id):
    with open('PeopleLocalization/data.csv') as f:
        df = pd.read_csv(f)
        room = df.loc[df['Image'] == id].iloc[0]['Room']
    return room


# print on map.png a red circle on the room passed as input
# useful for user visualization (this function will be used to display on GUI where a person is)
def print_on_map(room):
    map = cv2.imread('PeopleLocalization/images/map.png')
    new_map = cv2.resize(map, (400, 300))
    if room != '':
        center = map_grid[room]
        cv2.circle(new_map, center, 10, (0, 0, 255), -1)
    return new_map


''' How to use it
    if __name__ == "__main__":
        room = get_room("000.png")
        print_on_map(room)
'''


