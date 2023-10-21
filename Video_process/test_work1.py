import cv2

# Загрузка изображения
img = cv2.imread('Video_process/Images/frame0.jpg')

# Отображение изображения
cv2.namedWindow('image')
cv2.setMouseCallback('image', lambda event, x, y, flags, param: None)
while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if cv2.waitKey(1)== ord('q'):
        break
cv2.destroyAllWindows()

# Выделение части изображения
x, y, w, h = cv2.selectROI('image', img, fromCenter=False)

# Сохранение выделенной части в переменную
roi = img[y:y+h, x:x+w]

# Отображение выделенной части
# cv2.namedWindow('roi')
# cv2.setMouseCallback('roi', lambda event, x, y, flags, param: None)
# while True:
#     cv2.imshow('roi', roi)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
cv2.destroyAllWindows()