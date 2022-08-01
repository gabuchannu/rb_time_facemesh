import cv2
import numpy as np
import mediapipe as mp
import copy


class Facemesh:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.nose = {
            'x': [218, 438],
            'y': [168, 2]
        }

        self.rcheek = {
            'x': [100, 117, 118, 50, 36],
            'y': [100, 117, 118, 50, 36]
        }

        self.lcheek = {
            'x': [329, 346, 347, 280, 266],
            'y': [329, 346, 347, 280, 266],
        }

    def run(self, image):
        copy_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detect(image)

        if results is None: #顔検出できてなかったら
            return None

        face_points = results['face']
        nose_points = results['nose']
        lcheek_points = results['lcheek']
        rcheek_points = results['rcheek']

        face = self.clip(copy_image, face_points)
        nose = self.clip(copy_image, nose_points)
        lcheek = self.clip(copy_image, lcheek_points)
        rcheek = self.clip(copy_image, rcheek_points)

        self.draw_rect(image, face_points)
        self.draw_rect(image, nose_points)
        self.draw_rect(image, lcheek_points)
        self.draw_rect(image, rcheek_points)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return {
            'landmarks': image,
            'face': face,
            'nose': nose,
            'lcheek': lcheek,
            'rcheek': rcheek,
            'face_points': face_points,
            'nose_points': nose_points,
            'lcheek_points': lcheek_points,
            'rcheek_points': rcheek_points
        }

    def clip_and_draw(self, image, landmarks):
        copy_image = copy.deepcopy(image)

        face = self.clip(copy_image, landmarks['face'])
        nose = self.clip(copy_image, landmarks['nose'])
        lcheek = self.clip(copy_image, landmarks['lcheek'])
        rcheek = self.clip(copy_image, landmarks['rcheek'])

        self.draw_rect(image, landmarks['face'])
        self.draw_rect(image, landmarks['nose'])
        self.draw_rect(image, landmarks['lcheek'])
        self.draw_rect(image, landmarks['rcheek'])

        return {
            'image': image,
            'face': face,
            'nose': nose,
            'lcheek': lcheek,
            'rcheek': rcheek,
        }

    def detect(self, image):
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        image_width, image_height = image.shape[1], image.shape[0]

        if results.multi_face_landmarks is not None:
            if (len(results.multi_face_landmarks) == 1):
                landmarks = results.multi_face_landmarks[0]

                landmark_points = []
                for _, landmark in enumerate(landmarks.landmark):
                    landmark_x = min(
                        int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(
                        int(landmark.y * image_height), image_height - 1)

                    landmark_points.append((landmark_x, landmark_y))

                face = self.get_face(landmark_points)
                nose = self.get_part(landmark_points, points=self.nose)
                rcheek = self.get_part(landmark_points, points=self.rcheek)
                lcheek = self.get_part(landmark_points, points=self.lcheek)

                return {
                    'face': face,
                    'nose': nose,
                    'rcheek': rcheek,
                    'lcheek': lcheek,
                }

        return None

    def draw_rect(self, image, rect):
        cv2.rectangle(
            image,
            (rect[0], rect[2]),
            (rect[1], rect[3]),
            (2, 255, 0),
            1
        )

    def clip(self, image, rect):
        return image[
            rect[2]: rect[3],
            rect[0]: rect[1],
            :
        ]

    def get_face(self, landmark):
        np_landmark = np.array(landmark)

        xmax = np_landmark[:, 0].max()
        xmin = np_landmark[:, 0].min()
        ymax = np_landmark[:, 1].max()
        ymin = np_landmark[:, 1].min()

        return (xmin, xmax, ymin, ymax)

    def get_part(self, landmark, points):
        np_landmark = np.array(landmark)

        xmin = np.min(np_landmark[points['x'], 0])
        xmax = np.max(np_landmark[points['x'], 0])
        ymin = np.min(np_landmark[points['y'], 1])
        ymax = np.max(np_landmark[points['y'], 1])

        return (xmin, xmax, ymin, ymax)


if __name__ == '__main__':
    facemesh = Facemesh(0.7, 0.5)

    image = cv2.imread('./dog.jpg')
    results = facemesh.run(image)
    cv2.imwrite('./out2.png', results['landmarks'])
    cv2.imwrite('./out_face2.png', results['face'])
    cv2.imwrite('./out_nose2.png', results['nose'])
