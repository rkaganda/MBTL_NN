import config
from pathlib import Path
from os.path import isfile, join
from os import listdir
import dxcam
from PIL import Image
import tqdm


class StateImageCapture:
    def __init__(self, game_states):
        self.game_states = game_states
        self.existing_states = {}
        self.screen_grabber = dxcam.create()
        self.captured_states = {}
        self.last_frame = None
        self.last_frame_num = 0

        self.preload_existing_game_states()

    def preload_existing_game_states(self):
        # for each player
        for p in range(0, 2):
            self.existing_states[p] = []  # create array to store existing states
            path = Path("{}/state_img/{}".format(
                config.settings['data_path'], config.settings['p{}_model'.format(p)]['character']))
            path.mkdir(parents=True, exist_ok=True)

            # for each existing state file
            state_files = [f for f in listdir(path) if (isfile(join(path, f)) and f.startswith("state_"))]
            for f in state_files:
                self.existing_states[p].append(
                    f.replace("state_", "").replace(".jpg", "")
                )
            # create dict to store states while running
            self.captured_states[p] = {}

    def capture_new_states(self, frame_num):
        # out of bounds check
        frame_num = frame_num if frame_num < len(self.game_states) else len(self.game_states) - 1
        for p, p_states in self.game_states[frame_num]['game'].items():
            motion_frame = 0 if p_states['motion'] == 0 else 255 - p_states['motion']
            if "{}_{}".format(p_states['motion_type'], motion_frame) not in self.existing_states[p]:
                frame = self.screen_grabber.grab((0, 0, 1366, 768))
                if frame is None:  # if we already grabbed this frame
                    frame = self.last_frame
                else:
                    self.last_frame = frame
                self.captured_states[p]["{}_{}".format(p_states['motion_type'], motion_frame)] = frame
                self.existing_states[p].append("{}_{}".format(p_states['motion_type'], motion_frame))

    def store_new_states(self):
        for p, capped_states in self.captured_states.items():
            for state_name, frame in tqdm.tqdm(capped_states.items()):
                im = Image.fromarray(frame)
                path = Path("{}/state_img/{}".format(
                    config.settings['data_path'], config.settings['p{}_model'.format(p)]['character']))
                im.save("{}/state_{}.jpeg".format(path, state_name))
            capped_states.clear()