from app.face.face_recognition import predict


class FaceRecognitionService:
    @staticmethod
    def recognize_face(image_path: str) -> dict:
        """
        Predicts the identity of a face from a given image path.

        Args:
            image_path (str): Path to the image to recognize.

        Returns:
            dict: {
                "label": int,
                "name": str,
                "distance": float
            } or {
                "error": str
            }
        """
        try:
            label, name, distance = predict(image_path)
            return {
                "label": label,
                "name": name,
                "distance": round(float(distance), 2)
            }
        except Exception as e:
            return {
                "error": str(e)
            }
