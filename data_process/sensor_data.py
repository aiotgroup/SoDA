class SensorData(object):
    def __init__(self, user_id, action_id, attempt_id, accData, gyrData, label):
        self.user_id = user_id
        self.action_id = action_id
        self.attempt_id = attempt_id
        self.accData = accData
        self.gyrData = gyrData
        self.label = label
