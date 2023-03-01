class ActionScript:
    def __init__(self):
        self.current_frame = -1

        # forward then sweep
        self.action_script = ([40]*13) + \
                             ([32]*5) + \
                             ([40]*39) + \
                             [11] + \
                             ([40]*47)

    def get_current_frame(self):
        return self.current_frame

    def reset(self):
        self.current_frame = -1

    def get_action(self, states, frame):
        if self.current_frame == -1:
            if states[frame]['game'][0]['motion_type'] == 649:
                self.current_frame = 0
                return self.action_script[self.current_frame]
        elif states[frame]['game'][1]['hit'] == 1:
            self.reset()
        else:
            self.current_frame = self.current_frame+1
            if self.current_frame == len(self.action_script):
                self.current_frame = 0
                return self.action_script[self.current_frame]
            else:
                return self.action_script[self.current_frame]

        return 32


