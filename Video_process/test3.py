import cv2

# Загрузка изображения
img = cv2.imread('image.jpg')

# Отображение изображения
cv2.namedWindow('image')
cv2.setMouseCallback('image', lambda event, x, y, flags, param: None)
while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# Выделение части изображения
x, y, w, h = cv2.selectROI('image', img, fromCenter=False, showCrosshair=True)

# Сохранение выделенной части в переменную
roi = img[y:y+h, x:x+w]

# Отображение выделенной части
cv2.namedWindow('roi')
cv2.setMouseCallback('roi', lambda event, x, y, flags, param: None)

# Функция для изменения размера выделенной области
def resize_roi(roi, scale_x, scale_y):
    global roi, x, y, w, h
    if event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
        # Изменение размера выделенной области
            w = int(w * scale_x)
            h = int(h * scale_y)
            x = int(x * scale_x)
            y = int(y * scale_y)
            # Обрезка выделенной области до размеров изображения
            roi = roi[max(0, y):min(img.shape[0], y+h), max(0, x):min(img.shape[1], x+w)]

            # Создание переменных для изменения размера выделенной области
            scale_x = 1.0
            scale_y = 1.0

    # Отображение выделенной области
    while True:
        cv2.imshow('roi', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Изменение размера выделенной области при нажатии на клавишу '+' или '-'
    if cv2.waitKey(30) & 0xFF == ord('+'):
        scale_x += 0.1
        scale_y += 0.1
    elif cv2.waitKey(30) & 0xFF == ord('-'):
        scale_x -= 0.1
        scale_y -= 0.1
# Установка обработчика событий мыши для изменения размера выделенной области
cv2.setMouseCallback('roi', resize_roi)

# Закрытие окон
cv2.destroyAllWindows()