import cv2 as cv

def create_from_rgb(rgb_images, video_file_name):
    rgb_images = iter(rgb_images)
    img = next(rgb_images)

    codec = cv.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv.VideoWriter(video_file_name, codec, 30, (img.shape[1], img.shape[0]))

    for img in rgb_images:
        vid_writer.write(img)

    vid_writer.release()


def create_from_gray(gray_images, video_file_name):
    return create_from_rgb(
        map(lambda img: cv.cvtColor(img, cv.COLOR_GRAY2BGR), gray_images),
        video_file_name
    )


class _VideoPlayer:
    def __init__(self, reader, file_name):
        self.reader = reader
        self.file_name = file_name

    def step(self):
        ret, frame = self.reader.read()
        if not ret:
            self.reader.set(cv.CAP_PROP_POS_FRAMES, 0)
        else:
            cv.imshow(self.file_name, frame)

    def rewind(self):
        i = self.reader.get(cv.CAP_PROP_POS_FRAMES) - 2
        i = self.reader.get(cv.CAP_PROP_FRAME_COUNT) - 1 if i < 0 else i
        self.reader.set(cv.CAP_PROP_POS_FRAMES, i)
        self.step()

    def pause(self):
        print(f"paused at image {self.reader.get(cv.CAP_PROP_POS_FRAMES)}/{self.reader.get(cv.CAP_PROP_FRAME_COUNT)}")
        while True:
            key = cv.waitKey(30)
            if key == 32:  # Space => play
                return True
            elif key == 113:  # Q => quit
                return False
            elif key == 83 or key == 110:  # Right arrow or N => step
                self.step()
            elif key == 81 or key == 112:  # Left arrow or P => rewind
                self.rewind()

    def play(self):
        continue_playing = True
        while continue_playing:
            self.step()
            key = cv.waitKey(30) & 0xFF
            if key == 113:  # Q => quit
                continue_playing = False
            elif key == 32:  # Space => pause
                continue_playing = self.pause()


def play(video_file_name):
    video_reader = cv.VideoCapture(video_file_name)
    _VideoPlayer(video_reader, video_file_name).play()
    video_reader.release()
    cv.destroyAllWindows()
