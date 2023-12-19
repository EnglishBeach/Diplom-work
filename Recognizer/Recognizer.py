# %%
# %matplotlib qt
# %matplotlib ipympl

# %%
## Imports
print("Importing...")
import pandas as pd
from tqdm import tqdm

import cv2
import easyocr

from recognizer_modules import PreProcessor, PostProcessor, PathContainer

# %%
## Inputs
PATHS = PathContainer()
PATHS.print_paths()
VARIABLE_PATTERNS = {
    # 'Viscosity': r'-?\d{1,3}[\.,]\d',
    'Viscosity': r'-?\d{1,5}[\.,]?\d',
    'Temperature': r'-?\d{1,3}[\.,]\d',
}

# %%
## PreProcessor settings
class ImageProcessor(PreProcessor):
    Blur = range(1, 50)

    def process(self, image, gray_image=True):
        image = cv2.blur(image, (int(self['Blur']), int(self['Blur'])))
        if gray_image:
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            image = cv2.bitwise_not(image)

        return image


CAP = cv2.VideoCapture(PATHS.video_path)

FPS = int(CAP.get(cv2.CAP_PROP_FPS))
LENTH = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT) / FPS)
CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, START_FRAME = CAP.read()

processor = ImageProcessor([i for i in VARIABLE_PATTERNS])
print('Configurate image processing')

print(
    'Press:',
    '   Enter - save selection and continue',
    '   R     - reset video timer',
    '   Ecs/C - cancel selection',
    '   q/e   - time move',
    sep='\n',
)
configure = True
while configure:
    processor.configure_process(CAP)
    processor.select_window(CAP)
    processor.check_process(CAP)
    configure = False if input ('Continue (y or [n])? ')=='y' else True


# %%
## PostProcessor settings
class ValuePostProcessor(PostProcessor):

    def convert(self, value: str):
        value = value.replace(',', '.')
        try:
            result = float(value)
            return result
        except:
            return None

    @PostProcessor.check_type
    def image_sweep_check(self) -> list[str]:
        for i in range(1, 50):
            self.inner_processor['Blur'] = i
            processed_img = self.inner_processor(self.image)
            raw_value = [
                value for _, value, _ in self.reader.readtext(processed_img)
            ]
            result = self.isOK(raw_value)
            if result is not None: return result

    @PostProcessor.check_type
    def combine_check(self) -> list[str]:
        n_parts = len(self.input_value)
        combined_value =[]
        if n_parts == 1:
            value = self.input_value[0]
            combined_value = value[:3] + '.' + value[4:5]

        elif n_parts == 2:
            combined_value = '.'.join(self.input_value)

        elif n_parts == 3:
            combined_value = f'{self.input_value[0]}.{self.input_value[2]}'

        return self.isOK(combined_value)

input_fps = input('FPS to recognize: ')
try:
    read_fps = float(input_fps)
except:
    read_fps = 1

print('Starting recognizer...')
reader = easyocr.Reader(['en'])
checker = ValuePostProcessor(reader=reader, processor=processor)
# checker.active_checks_order =
# {check:checker.all_checks[check]
# for check in ['inner_processor_check','value_combine']}
print('Active checks:\n', [i for i in checker.active_checks_order])

# %%
## Recognize
print('Recognizing:')
errors = 0
frame_line = tqdm(iterable=range(0, FPS * LENTH, int(FPS / read_fps)))
frame_line.set_description(f'Errors: {errors: >4}')
DATA = []

for i_frame in frame_line:
    CAP.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    _, frame = CAP.read()
    i_text = {'time': round(i_frame / FPS, 1)}
    processed_frame = processor(frame)
    stricted_images = processor.strict(processed_frame)

    for var, pattern in VARIABLE_PATTERNS.items():
        var_image = stricted_images[var]
        raw_value = [value for _, value, _ in reader.readtext(var_image)]

        verbose, result = checker.check(
            input_value=raw_value,
            pattern=pattern,
            image = var_image,
            )

        # if mark == 'error':
        #     # processor.configure_process(CAP,start_frame=i_frame)
        #     processor.select_window(CAP,start_frame=i_frame)
        #     # processor.check_process(CAP,start_frame=i_frame)
        #     checker.reload_processor(processor)
        #     mark, result = checker.check(image=var_image,
        #                 raw_value=raw_value,
        #                 rules=rules)
        #     mark= f'*{mark}'

        i_text[var] = result
        i_text[var + '_verbose'] = verbose

    if None in i_text.values():
        errors += 1
        frame_line.set_description(f'{errors: >4} errors')
    DATA.append(i_text)

# %%
## Saving
df = pd.DataFrame(DATA)
df.dropna().to_csv(PATHS.data_path,index=0)
