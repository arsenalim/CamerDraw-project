from cv2 import cv2
import mediapipe as mp
import numpy as np
import math

'''Функция get_points возвращает список всех ключевых точек кисти, конвертированных в пиксели
   The get_points function returns a list of all key brush points converted to pixels'''

def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


'''Функция palm_size находит расстояние между точками 0 и 5 (начало кисти и начало указательного пальца), 
   для определения кулака
   The palm_size function finds the distance between points 0 and 5 
   (the beginning of the hand and the beginning of the index finger) to determine the fist'''


def palm_size(landmark, shape):
    x_1, y_1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x_2, y_2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2) ** 0.5


'''Функция de_size конвертирует координаты изображения из частей (от 0 да 1) в количество пикселей
   The function de_size converts the coordinates of the image from parts (from 0 to 1) to the number of pixels'''


def de_size(landmark, no_of_point, shape):
    x_1, y_1 = landmark[no_of_point].x * shape[1], landmark[no_of_point].y * shape[0]
    return x_1, y_1


# открытие детектора и все настройки
# opening the detector and all settings

color = (0, 255, 0)            # настроить цвет линии / adjust the line color
width_px = 7                   # настроить ширину линии / adjust the line width

handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
drawings = [[]]
to_append_new = True

# главный цикл
# main cycle

while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    # конвертирование для обработки изображения
    # image conversion for processing

    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)

    if results.multi_hand_landmarks is not None:

        right = 0

        # распознание двух пальцев
        # two finger recognition

        x1_IF, y1_IF = de_size(results.multi_hand_landmarks[0].landmark, 8, flippedRGB.shape)
        x2_IF, y2_IF = de_size(results.multi_hand_landmarks[0].landmark, 5, flippedRGB.shape)

        index_finger = math.hypot(x1_IF - x2_IF, y1_IF - y2_IF)

        x_MF, y_MF = de_size(results.multi_hand_landmarks[0].landmark, 12, flippedRGB.shape)
        distance = math.hypot(x_MF - x1_IF, y_MF - y1_IF)

        two_fingers = distance > index_finger / 3

        # если пальцы не близко то добавляем новую точку
        # if the fingers are not close, then add a new point

        if two_fingers:
            x = int(results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[1])
            y = int(results.multi_hand_landmarks[0].landmark[8].y * flippedRGB.shape[0])

            drawings[-1].append((x, y))
            to_append_new = True

        else:
            if to_append_new:
                drawings.append([])
                to_append_new = False

        # распознание кулака
        # fist recognition

        (x_, y_), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)

        fist = 2 * r / ws > 1.4

        if not fist:
            drawings = [[]]

    else:
        if to_append_new:
            drawings.append([])
            to_append_new = False

    # рисование всех контуров
    # drawing all contours

    for figure in drawings:
        if len(figure) > 1:
            for line in range(1, len(figure)):
                x1 = figure[line][0]
                y1 = figure[line][1]
                x2 = figure[line - 1][0]
                y2 = figure[line - 1][1]

                cv2.line(flippedRGB, (x1, y1), (x2, y2), color, width_px)

    # конвертирование обратно и вывод изображения
    # converting back and outputting the image

    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("CamerDraw", res_image)

handsDetector.close()
