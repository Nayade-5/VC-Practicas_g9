from cvzone.HandTrackingModule import HandDetector
import cv2

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, track_con=0.5, model_complexity=1):
        # cvzone HandDetector args: staticMode, maxHands, modelComplexity, detectionCon, minTrackCon
        self.detector = HandDetector(staticMode=mode, maxHands=max_hands, modelComplexity=model_complexity, detectionCon=detection_con, minTrackCon=track_con)
        self.hands = []

    def find_hands(self, img, draw=True):
        # cvzone returns (hands, img)
        self.hands, img = self.detector.findHands(img, draw=draw)
        return img

    def find_position(self, img, hand_no=0):
        # Returns list in format [id, x, y] to match previous behavior
        if self.hands:
            if hand_no < len(self.hands):
                my_hand = self.hands[hand_no]
                lm_list_cvzone = my_hand["lmList"]
                # Convert to old format [id, x, y]
                lm_list = []
                for i, lm in enumerate(lm_list_cvzone):
                    # cvzone lmList is [x, y, z]
                    lm_list.append([i, lm[0], lm[1]])
                return lm_list
        return []

    def get_fingers(self, hand_no=0):
        if self.hands and hand_no < len(self.hands):
            return self.detector.fingersUp(self.hands[hand_no])
        return []
