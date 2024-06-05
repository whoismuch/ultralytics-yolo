import bentoml
from bentoml.io import Image
from bentoml.io import PandasDataFrame
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import time

class Yolov5Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import torch

        self.session = ort.InferenceSession('best.onnx', providers=["CPUExecutionProvider"])

        # Get the model inputs
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # if torch.cuda.is_available():
        #     self.session.cuda()
        # else:
        #     self.session.cpu()

        # Config inference settings
        self.inference_size = 640

    def preprocess(self, image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        open_cv_image = np.array(image)
        # Преобразуем RGB в BGR
        self.img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))
        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0
        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        # Return the preprocessed image data
        return image_data


    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= 0.5:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)

        results = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            results.append((box, class_id))

            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return results

    @bentoml.Runnable.method()
    def inference(self, input_img):
        # Preprocess the image data
        img_data = self.preprocess(input_img)
        start_time = time.time()
        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})
        end_time = time.time()
        inference_time = end_time - start_time

        # Perform post-processing on the outputs to obtain output image.
        results = self.postprocess(outputs)

        print(results)
        # Преобразуем данные в DataFrame
        if results != []:
            df = pd.DataFrame(results, columns=['bbox', 'class'])
            df['inference_time'] = inference_time
        else:
            df = pd.DataFrame([inference_time], columns=['inference_time'])
        return df


yolo_v5_runner = bentoml.Runner(Yolov5Runnable, max_batch_size=30)

svc = bentoml.Service("yolo_v5_demo", runners=[yolo_v5_runner])

@svc.api(input=Image(), output=(PandasDataFrame()))
async def invocation(input_img):
    batch_ret = await yolo_v5_runner.inference.async_run(input_img)
    return batch_ret


