import cv2
import glob
import _pickle as pickle

def main():
    names = glob.glob("paintings_db/*.png")
    p = []
    orb = cv2.ORB_create()
    for name in names:
        im = cv2.imread(name)
        kp, des = orb.detectAndCompute(im, None)
        points = []
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave,
                    point.class_id)
            points.append(temp)
        p.append({'im': im, 'filename': name, 'kp': points, 'des': des})

    with open('paintings_db/db_paintings.pickle', 'wb') as db_paintings_file:
        pickle.dump(p, db_paintings_file)


if __name__ == '__main__':
    main()